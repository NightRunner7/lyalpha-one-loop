#!/usr/bin/env python3
"""Read-only preflight for existing accDM fit inputs. This is NOT a fit runner.

Default: inspect JSON/NPZ, independently reproduce the recorded bundle digest,
check DR12 release files, saved bounds and seed provenance. No CLASS, fitting,
parameter clipping or changes to input files. --verify-chi2 additionally uses
the audited canonical loader from the separately installed profiling package.
Only reports in the explicit --out directory are written.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Canonical imports must not leave __pycache__ files in the source project.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SUPPORTED_FIT_SHA256 = '845ec6896c8901df8535d2bc8069344842235e1fc38ba4bca7fd5a59acfb30b6'
NAMES = ('log_alpha_F','beta_F','alpha_bias','beta_bias','alpha_ct','beta_ct')
OLD_ROOTS = (Path('/home/2/ks405818/Master/lyalpha_one_loop'),
             Path('/home/krzysztof/Workspace/lyalpha_one_loop'))
ARRAYS = ('z','k_input','p_total_input','p_loop_input','k_loop','p_tree',
          'p22','p13','p13_error','channels_one_loop','hubble_km_s_mpc','velocity_to_hmpc')
SCALARS = ('h','omega_m','loop_weight')
ATOL, RTOL = 1e-4, 1e-7


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def present(value):
    return value is not None and not (isinstance(value,(float,np.floating)) and np.isnan(value))


def key_of(m,f):
    values=[float(m),float(f)]
    if not np.isfinite(values).all():raise ValueError('Nonfinite native coordinates')
    return '|'.join('0.0000000000' if abs(x)<.5e-10 else f'{x:.10f}' for x in values)


def resolve(raw,root):
    if not present(raw) or str(raw).strip().lower() in {'','none','nan'}:
        raise ValueError('Missing path')
    p=Path(str(raw)).expanduser()
    if not p.is_absolute():return (root/p).resolve()
    for previous in OLD_ROOTS:
        try:relative=p.relative_to(previous)
        except ValueError:continue
        return (root/relative).resolve()
    return p.resolve()


def inspect_npz(path):
    """Exact field order/hash from audited TheoryBundle.bundle_digest()."""
    with np.load(path,allow_pickle=False) as data:
        missing=set(ARRAYS+SCALARS+('metadata_json',))-set(data.files)
        if missing:raise ValueError(f'Missing NPZ fields: {sorted(missing)}')
        meta=json.loads(str(data['metadata_json'].item()))
        if not isinstance(meta,dict):raise ValueError('metadata_json is not an object')
        arrays={name:np.asarray(data[name]) for name in ARRAYS}
        scalars={name:float(data[name].item()) for name in SCALARS}
    z,ki,kl=(arrays[name] for name in ('z','k_input','k_loop'))
    for name,grid in [('z',z),('k_input',ki),('k_loop',kl)]:
        if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0):
            raise ValueError(f'Invalid strictly increasing {name} grid')
        if name!='z' and np.any(grid<=0):raise ValueError(f'Nonpositive {name}')
    nz,ni,nk=len(z),len(ki),len(kl)
    shapes={'p_total_input':(nz,ni),'p_loop_input':(nz,ni),'p_tree':(nz,nk),
            **{n:(nz,nk,3) for n in ('p22','p13','p13_error','channels_one_loop')},
            'hubble_km_s_mpc':(nz,),'velocity_to_hmpc':(nz,)}
    for name,shape in shapes.items():
        if arrays[name].shape!=shape:raise ValueError(f'{name}: {arrays[name].shape} != {shape}')
        if not np.isfinite(arrays[name]).all():raise ValueError(f'Nonfinite {name}')
    if not np.isfinite(list(scalars.values())).all():raise ValueError('Nonfinite scalar')
    digest=hashlib.sha256()
    for name in ARRAYS:
        a=np.ascontiguousarray(arrays[name])
        digest.update(str(a.dtype).encode())
        digest.update(np.asarray(a.shape,dtype=np.int64).tobytes())
        digest.update(a.tobytes())
    for name in SCALARS:digest.update(np.float64(scalars[name]).tobytes())
    actual=digest.hexdigest()
    if not meta.get('bundle_digest') or meta['bundle_digest']!=actual:
        raise ValueError('NPZ bundle_digest does not match the actual arrays/scalars')
    # Native coordinates come from the CSV/fit identity; no physical conversion
    # or campaign-name inference of resolution is performed by this audit.
    return dict(metadata=meta,bundle_digest=actual,z=z.tolist(),k_min=float(kl[0]),
                k_max=float(kl[-1]),n_k=nk,scalars=scalars)


class Audit:
    def __init__(self,args):
        self.args=args;self.root=args.project_root.expanduser().resolve()
        self.issues=[];self.missing=[];self.files={};self.point_rows=[];self.seed_rows=[]
        self.data_dirs={};self.seed_best={};self.baseline={}
        self.fit_hash=None;self.code_ok=False;self.canonical_loader=None

    def issue(self,code,message,*,target='',path='',severity='error'):
        self.issues.append(dict(target=target,severity=severity,code=code,path=str(path),message=str(message)))

    def file(self,path,kind,target=''):
        path=Path(path)
        if not path.is_file():
            self.issue('missing_file',f'Missing {kind}',target=target,path=path)
            self.missing.append(dict(target=target,kind=kind,path=str(path)))
            return False
        try:self.files[str(path)]=dict(path=str(path),kind=kind,size_bytes=path.stat().st_size)
        except OSError as exc:self.issue('file_stat_failed',exc,target=target,path=path);return False
        return True

    def table(self,path,role):
        if not self.file(path,role):return pd.DataFrame()
        try:
            frame=pd.read_csv(path,dtype={'coordinate_key':str,'target':str})
            if frame.empty:raise ValueError('Empty table')
            return frame
        except Exception as exc:self.issue('unreadable_table',exc,path=path);return pd.DataFrame()

    def setup(self):
        fit=self.root/'lyalpha_pt/fit.py'
        if self.file(fit,'canonical_fit_source'):
            try:
                self.fit_hash=sha256(fit);self.code_ok=self.fit_hash==SUPPORTED_FIT_SHA256
                if not self.code_ok:self.issue('unsupported_fit_source','fit.py hash differs from audited model; do not bypass this check',path=fit)
            except Exception as exc:self.issue('source_read_failed',exc,path=fit)
        baseline_path=HERE/'baseline_full_snapshot.csv'
        baseline=self.table(baseline_path,'frozen_full_baseline')
        if not baseline.empty:
            if 'coordinate_key' not in baseline:self.issue('baseline_schema','Missing coordinate_key',path=baseline_path)
            elif baseline.coordinate_key.duplicated().any():self.issue('baseline_duplicates','Duplicate baseline identities',path=baseline_path)
            else:self.baseline={str(r['coordinate_key']):r for r in baseline.to_dict('records')}
        if self.args.verify_chi2:
            if not self.code_ok:
                self.issue('canonical_verification_blocked','--verify-chi2 blocked by missing/unsupported canonical fit.py')
            else:
                package=self.root/'accdm_reprofile_30_points'
                module_path=package/'accdm_reprofile/inputs.py'
                if self.file(module_path,'canonical_context_adapter'):
                    try:
                        sys.path.insert(0,str(package));sys.path.insert(0,str(self.root))
                        module=importlib.import_module('accdm_reprofile.inputs')
                        if Path(module.__file__).resolve()!=module_path.resolve():raise ValueError('Imported context adapter from another directory')
                        self.canonical_loader=module.load_context
                    except Exception as exc:self.issue('canonical_adapter_import_failed',exc,path=module_path)

    def read_seeds(self):
        directory=self.args.seed_dir.expanduser().resolve()
        best_path=directory/'seed_best_points.csv'
        attempts_path=directory/'seed_attempts.csv'
        # --seed-dir may also point directly to a current reprofile output.
        if not best_path.exists() and (directory/'reprofile_best_points.csv').is_file():best_path=directory/'reprofile_best_points.csv'
        if not attempts_path.exists() and (directory/'reprofile_attempts.csv').is_file():attempts_path=directory/'reprofile_attempts.csv'
        best=self.table(best_path,'seed_best_points');attempts=self.table(attempts_path,'seed_attempts')
        required={'target','chi2_old','chi2_best','theory_digest'}|{'parameter_best_'+n for n in NAMES}
        if not best.empty:
            absent=required-set(best)
            if absent:self.issue('seed_best_schema',f'Missing columns: {sorted(absent)}',path=best_path)
            elif best.target.duplicated().any():self.issue('seed_best_duplicate_targets','Duplicate seed targets',path=best_path)
            else:
                for row in best.to_dict('records'):
                    key=str(row['target']);before=len(self.issues)
                    try:
                        old=self.baseline.get(key)
                        if old is None:raise ValueError('Seed target absent from frozen full baseline')
                        if 'log10m_acc' in row and 'log10f_acc' in row and key_of(row['log10m_acc'],row['log10f_acc'])!=key:raise ValueError('Seed native coordinates differ from target')
                        if row['theory_digest']!=old.get('theory_digest'):raise ValueError('Seed/baseline theory_digest mismatch')
                        if not np.isclose(float(row['chi2_old']),float(old['chi2_one_loop']),atol=ATOL,rtol=RTOL):raise ValueError('Seed old chi2 differs from baseline')
                        theta=np.array([row['parameter_best_'+n] for n in NAMES],float)
                        bounds=np.array([[old['lower_'+n],old['upper_'+n]] for n in NAMES],float)
                        if not np.isfinite(theta).all() or np.any(theta<bounds[:,0]) or np.any(theta>bounds[:,1]):raise ValueError('Seed theta nonfinite or outside baseline bounds')
                        if not np.isfinite(float(row['chi2_best'])):raise ValueError('Nonfinite seed chi2')
                        for n in NAMES:
                            if 'parameter_old_'+n in row and not np.isclose(float(row['parameter_old_'+n]),float(old['parameter_'+n]),atol=0,rtol=1e-9):raise ValueError(f'Seed old {n} differs from baseline')
                            for side in ('lower','upper'):
                                if side+'_'+n in row and not np.isclose(float(row[side+'_'+n]),float(old[side+'_'+n]),atol=1e-12,rtol=1e-12):raise ValueError(f'Seed {side} bound for {n} differs')
                        self.seed_best[key]=dict(row,theta=theta.tolist())
                    except Exception as exc:self.issue('seed_best_inconsistent',exc,target=key,path=best_path)
                    self.seed_rows.append(dict(target=key,kind='best',status='metadata_consistent' if len(self.issues)==before else 'inconsistent',canonical_seed_verified=False))
        attempt_required={'target','trial_id','theta_selected','chi2_selected','chi2_old'}
        if not attempts.empty:
            absent=attempt_required-set(attempts)
            if absent:self.issue('seed_attempt_schema',f'Missing columns: {sorted(absent)}',path=attempts_path)
            else:
                if attempts.trial_id.duplicated().any():self.issue('seed_attempt_duplicates','Duplicate trial_id values',path=attempts_path)
                for row in attempts.to_dict('records'):
                    key=str(row['target']);before=len(self.issues)
                    try:
                        if key not in self.seed_best:raise ValueError('No consistent best-seed identity/digest for this attempt target')
                        old=self.baseline[key];seed=self.seed_best[key]
                        theta=np.asarray(json.loads(row['theta_selected']) if isinstance(row['theta_selected'],str) else row['theta_selected'],float)
                        bounds=np.array([[old['lower_'+n],old['upper_'+n]] for n in NAMES],float)
                        if theta.shape!=(6,) or not np.isfinite(theta).all() or np.any(theta<bounds[:,0]) or np.any(theta>bounds[:,1]):raise ValueError('Attempt selected theta invalid/outside bounds')
                        for i,n in enumerate(NAMES):
                            if 'parameter_'+n in row and not np.isclose(float(row['parameter_'+n]),theta[i],atol=0,rtol=1e-9):raise ValueError(f'Attempt selected theta vs scalar column mismatch: {n}')
                        if not np.isfinite(float(row['chi2_selected'])):raise ValueError('Nonfinite attempt chi2_selected')
                        if not np.isclose(float(row['chi2_old']),float(old['chi2_one_loop']),atol=ATOL,rtol=RTOL):raise ValueError('Attempt old chi2 differs from baseline')
                        if float(seed['chi2_best'])>float(row['chi2_selected'])+ATOL+RTOL*abs(float(row['chi2_selected'])):raise ValueError('Best-seed table omits a lower selected attempt')
                        if present(row.get('theory_digest')) and row['theory_digest']!=seed['theory_digest']:raise ValueError('Attempt theory digest mismatch')
                    except Exception as exc:self.issue('seed_attempt_inconsistent',exc,target=key,path=attempts_path)
                    self.seed_rows.append(dict(target=key,kind='attempt',trial_id=row.get('trial_id'),status='metadata_consistent' if len(self.issues)==before else 'inconsistent',canonical_seed_verified=False))

    def check_dr12(self,directory):
        directory=Path(directory)
        if str(directory) in self.data_dirs:return self.data_dirs[str(directory)]
        result=dict(path=str(directory),status='checked',files={})
        arrays={}
        for name,shape in [('Pk1D_data.dat',(455,6)),('Pk1D_syst.dat',(455,8)),('Pk1D_cor.dat',(455,35))]:
            path=directory/name
            if not self.file(path,'DR12_'+name):result['status']='failed';continue
            try:
                value=np.loadtxt(path)
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError(f'Expected finite array {shape}, found {value.shape}')
                arrays[name]=value;result['files'][name]=dict(sha256=sha256(path),shape=list(value.shape),size_bytes=path.stat().st_size)
            except Exception as exc:self.issue('invalid_dr12_file',exc,path=path);result['status']='failed'
        if len(arrays)==3:
            try:
                d=arrays['Pk1D_data.dat'];z,counts=np.unique(d[:,0],return_counts=True)
                if len(z)!=13 or not np.all(counts==35):raise ValueError('Expected 13 redshift blocks of 35 points')
                if np.any(d[:,1]<=0) or np.any(d[:,3]<=0):raise ValueError('k_velocity/statistical errors must be positive')
                corr=arrays['Pk1D_cor.dat'].reshape(13,35,35)
                if not np.allclose(np.diagonal(corr,axis1=1,axis2=2),1.,atol=1e-8):raise ValueError('Correlation block diagonal differs from one')
                mask=(d[:,0]>=3.-1e-10)&(d[:,0]<=4.2+1e-10)
                if int(mask.sum())!=245 or len(np.unique(d[mask,0]))!=7:raise ValueError('Fiducial selection is not 245 points/7 redshifts')
                result['selected_points']=245;result['selected_redshifts']=np.unique(d[mask,0]).tolist()
            except Exception as exc:self.issue('invalid_dr12_layout',exc,path=directory);result['status']='failed'
        self.data_dirs[str(directory)]=result
        return result

    def point(self,row,role):
        result=dict(role=role,coordinate_key=str(row.get('coordinate_key','')),canonical_chi2_status='not_requested')
        key=result['coordinate_key'];start=len(self.issues);record=None;info=None;fp=None;tp=None;data_dir=None
        try:
            if key!=key_of(row['log10m_acc'],row['log10f_acc']):raise ValueError('coordinate_key differs from native coordinates')
            old=self.baseline.get(key)
            if old is None:raise ValueError('Target not present in frozen full baseline')
            for field in ('theory_digest','covariance_mode'):
                if row.get(field)!=old.get(field):raise ValueError(f'Manifest {field} differs from full baseline')
            for field in ['chi2_one_loop','k_uv_cut_hmpc']+['parameter_'+n for n in NAMES]+[s+'_'+n for s in ('lower','upper') for n in NAMES]:
                a,b=float(row[field]),float(old[field])
                if not np.isfinite([a,b]).all() or not np.isclose(a,b,atol=0,rtol=1e-9):raise ValueError(f'Manifest {field} differs from full baseline')
        except Exception as exc:self.issue('manifest_identity',exc,target=key)
        for field,kind in [('fit_path','fit_json'),('theory_path','theory_npz')]:
            try:
                p=resolve(row.get(field),self.root);result[field]=str(p)
                if self.root!=p and self.root not in p.parents:self.issue('external_absolute_path','Resolved path is outside --project-root; review this explicit source',target=key,path=p,severity='warning')
                if self.file(p,kind,key):
                    result[kind+'_available']=True;result[kind+'_size_bytes']=p.stat().st_size
                    if kind=='fit_json':fp=p
                    else:tp=p
                else:result[kind+'_available']=False
            except Exception as exc:self.issue('path_resolution',exc,target=key,path=str(row.get(field)));result[kind+'_available']=False
        if fp is not None:
            try:
                payload=json.loads(fp.read_text(encoding='utf-8'));record=payload['fits']['one_loop']
                result['fit_sha256']=sha256(fp)
                if record.get('mode','one_loop')!='one_loop':raise ValueError('Saved fit mode differs')
                if resolve(payload['theory_file'],self.root)!=resolve(row['theory_path'],self.root):raise ValueError('Fit theory_file and manifest theory_path differ after remapping')
                data_dir=resolve(self.args.data_dir if self.args.data_dir is not None else payload.get('data_dir'),self.root)
                result['data_dir']=str(data_dir)
                data_check=self.check_dr12(data_dir)
                if data_check['status']=='failed':self.issue('point_dr12_invalid','Referenced DR12 directory failed file/layout checks',target=key,path=data_dir)
                if not np.isclose(float(record['chi2']),float(row['chi2_one_loop']),atol=ATOL,rtol=RTOL):raise ValueError('Fit chi2 differs from manifest')
                if record['covariance_mode']!=row['covariance_mode']:raise ValueError('Fit covariance_mode differs')
                if not np.isclose(float(record['k_uv_cut_hmpc']),float(row['k_uv_cut_hmpc']),atol=1e-12,rtol=1e-12):raise ValueError('Fit UV cutoff differs')
                if record['theory_digest']!=row['theory_digest']:raise ValueError('Fit theory_digest differs')
                if int(record['n_data'])!=245:raise ValueError('Fit n_data is not fiducial 245')
                diagnostics=record['boundary_diagnostics'];names=[d['parameter'] for d in diagnostics]
                if len(names)!=len(set(names)) or not set(NAMES).issubset(names):raise ValueError('Missing/duplicate nuisance bounds')
                by={d['parameter']:d for d in diagnostics};bounds={}
                for n in NAMES:
                    value=float(record['parameters'][n]);lo=float(by[n]['lower']);hi=float(by[n]['upper'])
                    if not np.isfinite([value,lo,hi]).all() or lo>=hi or value<lo-1e-10 or value>hi+1e-10:raise ValueError(f'Invalid saved value/bounds: {n}')
                    if not np.isclose(value,float(row['parameter_'+n]),atol=0,rtol=1e-9):raise ValueError(f'Fit nuisance differs from manifest: {n}')
                    for side,x in [('lower',lo),('upper',hi)]:
                        if not np.isclose(x,float(row[side+'_'+n]),atol=1e-12,rtol=1e-12):raise ValueError(f'Fit {side} bound differs: {n}')
                    bounds[n]=[lo,hi]
                result['bounds_json']=json.dumps(bounds,sort_keys=True)
                result['chi2_saved']=float(record['chi2']);result['covariance_mode']=record['covariance_mode'];result['k_uv_cut_hmpc']=float(record['k_uv_cut_hmpc'])
            except Exception as exc:self.issue('fit_json_inconsistent',exc,target=key,path=fp)
        if tp is not None:
            try:
                info=inspect_npz(tp);meta=info['metadata'];result['theory_sha256']=sha256(tp)
                result.update(bundle_digest=info['bundle_digest'],n_k=info['n_k'],k_min=info['k_min'],k_max=info['k_max'],
                    redshifts_json=json.dumps(info['z']),actual_class_params_json=json.dumps(meta.get('class_params',{}),sort_keys=True),
                    numerics_json=json.dumps(meta.get('numerics',{}),sort_keys=True),model_json=json.dumps(meta.get('model',{}),sort_keys=True),
                    loop_source=meta.get('loop_source'),loop_weight=info['scalars']['loop_weight'])
                if info['bundle_digest']!=row.get('theory_digest'):raise ValueError('Actual NPZ digest differs from manifest')
                if record is not None and info['bundle_digest']!=record.get('theory_digest'):raise ValueError('Actual NPZ digest differs from fit JSON')
                if len(info['z'])!=7 or not np.allclose(info['z'],np.arange(3.,4.21,.2),atol=1e-9,rtol=0):raise ValueError('NPZ redshifts differ from fiducial DR12 selection')
                cutoff=float(row['k_uv_cut_hmpc'])
                if not (info['k_min']<cutoff<=info['k_max']*(1+1e-12)):raise ValueError('Fit cutoff outside saved theory coverage')
                if 'class_params' not in meta:self.issue('missing_actual_class_metadata','No top-level actual CLASS settings; do not infer precision from campaign label',target=key,path=tp,severity='warning')
            except Exception as exc:self.issue('theory_npz_inconsistent',exc,target=key,path=tp)
        point_errors=[x for x in self.issues[start:] if x['severity']=='error']
        result['metadata_status']='passed' if not point_errors else 'failed'
        if self.args.verify_chi2:
            result['canonical_chi2_status']='blocked'
            if self.canonical_loader is None:
                self.issue('point_canonical_blocked','Canonical source/adapter gate failed',target=key)
            elif point_errors:self.issue('point_canonical_blocked','Point metadata/files failed preflight',target=key)
            else:
                ctx=None
                try:
                    ctx=self.canonical_loader(self.root,row,self.args.data_dir)
                    if ctx['source_sha256']!=SUPPORTED_FIT_SHA256:raise ValueError('Canonical fitter changed after source gate')
                    result['canonical_chi2_status']='verified';result['chi2_recomputed']=ctx['chi2_recomputed'];result['chi2_difference']=ctx['chi2_recomputed']-ctx['chi2_saved']
                    if key in self.seed_best:
                        seed=self.seed_best[key];actual=float(ctx['fitter'].chi2(seed['theta']))
                        if not np.isclose(actual,float(seed['chi2_best']),atol=ATOL,rtol=RTOL):raise ValueError('Best seed chi2 does not reproduce on its exact vector')
                        result['seed_chi2_recomputed']=actual
                        for s in self.seed_rows:
                            if s['target']==key and s['kind']=='best':s['canonical_seed_verified']=True
                except Exception as exc:
                    result['canonical_chi2_status']='failed';self.issue('canonical_chi2_failed',exc,target=key)
                finally:
                    del ctx
                    gc.collect()
        result['issue_count']=len(self.issues)-start;self.point_rows.append(result)

    def run(self):
        self.setup();self.read_seeds();targets=[]
        for path,role in [(self.args.manifest,'upper_grid'),(self.args.controls,'control')]:
            table=self.table(path,role+'_manifest')
            if table.empty:continue
            required={'coordinate_key','log10m_acc','log10f_acc','fit_path','theory_path','chi2_one_loop','theory_digest','covariance_mode','k_uv_cut_hmpc'}|{p+n for p in ('parameter_','lower_','upper_') for n in NAMES}
            absent=required-set(table)
            if absent:self.issue('manifest_schema',f'Missing columns {sorted(absent)}',path=path);continue
            if table.coordinate_key.duplicated().any():self.issue('manifest_duplicates','Duplicate coordinate keys',path=path)
            targets.extend((r,role) for r in table.to_dict('records'))
        seen=set()
        for i,(row,role) in enumerate(targets):
            key=row['coordinate_key']
            if key in seen:self.issue('cross_manifest_duplicate','Coordinate appears in multiple requested roles; audited once',target=key,severity='warning');continue
            seen.add(key)
            try:self.point(row,role)
            except Exception as exc:
                self.issue('unexpected_point_error',f'{type(exc).__name__}: {exc}',target=key)
                self.point_rows.append(dict(coordinate_key=key,role=role,metadata_status='failed',canonical_chi2_status='failed' if self.args.verify_chi2 else 'not_requested'))
            if (i+1)%50==0:print(f'Audited {i+1}/{len(targets)} points',flush=True)
        if not self.data_dirs:self.check_dr12(resolve(self.args.data_dir if self.args.data_dir is not None else 'data',self.root))
        output=self.args.out.expanduser().resolve()
        report_names=('all_issues.csv','missing_files.csv','preflight_points.csv','seed_audit.csv','file_inventory.csv','preflight_summary.json')
        collisions=[str(output/name) for name in report_names if str((output/name).resolve()) in self.files]
        if collisions:
            print(json.dumps(dict(error='Output report path would overwrite an input; choose a different --out',paths=collisions)),file=sys.stderr)
            return 2
        output.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(self.issues,columns=['target','severity','code','path','message']).to_csv(output/'all_issues.csv',index=False)
        pd.DataFrame(self.missing,columns=['target','kind','path']).to_csv(output/'missing_files.csv',index=False)
        point_columns=['role','coordinate_key','metadata_status','canonical_chi2_status']
        points=pd.DataFrame(self.point_rows) if self.point_rows else pd.DataFrame(columns=point_columns)
        points.to_csv(output/'preflight_points.csv',index=False)
        pd.DataFrame(self.seed_rows).to_csv(output/'seed_audit.csv',index=False)
        pd.DataFrame(self.files.values(),columns=['path','kind','size_bytes']).to_csv(output/'file_inventory.csv',index=False)
        errors=sum(x['severity']=='error' for x in self.issues)
        verified=sum(x.get('canonical_chi2_status')=='verified' for x in self.point_rows)
        summary=dict(audit_kind='READ_ONLY_INPUT_PREFLIGHT_NOT_FIT_RUNNER',created_utc=datetime.now(timezone.utc).isoformat(),
            project_root=str(self.root),requested_manifests=[str(self.args.manifest),str(self.args.controls)],
            supported_fit_sha256=SUPPORTED_FIT_SHA256,actual_fit_sha256=self.fit_hash,canonical_code_supported=self.code_ok,
            canonical_chi2_requested=self.args.verify_chi2,canonical_chi2_verified_points=verified,
            n_points=len(self.point_rows),metadata_passed_points=sum(r.get('metadata_status')=='passed' for r in self.point_rows),
            fit_files_available=sum(bool(r.get('fit_json_available')) for r in self.point_rows),
            theory_files_available=sum(bool(r.get('theory_npz_available')) for r in self.point_rows),
            missing_file_records=len(self.missing),errors=errors,warnings=len(self.issues)-errors,
            unique_input_files=len(self.files),unique_input_bytes=sum(v['size_bytes'] for v in self.files.values()),
            data_directories=self.data_dirs,seed_best_consistent=len(self.seed_best),
            seed_best_canonical_verified=sum(r['kind']=='best' and r.get('canonical_seed_verified',False) for r in self.seed_rows),
            actual_metadata_and_bounds_report='preflight_points.csv',
            actual_bounds_groups=[dict(bounds=json.loads(value),n_points=int(count))
                for value,count in points.get('bounds_json',pd.Series(dtype=str)).dropna().value_counts().items()],
            actual_numerics_groups=[dict(numerics=json.loads(value),n_points=int(count))
                for value,count in points.get('numerics_json',pd.Series(dtype=str)).dropna().value_counts().items()],
            seeds_are_candidates_only=True,baseline_was_overwritten=False,input_files_modified=False,
            structural_preflight_passed=errors==0 and bool(self.point_rows),
            ready_for_reprofile=bool(self.args.verify_chi2 and errors==0 and self.point_rows and verified==len(self.point_rows)),
            note='Metadata inspection does not establish optimizer convergence or validity of the cosmological model. Without --verify-chi2 no likelihood reproduction is claimed.')
        (output/'preflight_summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ('n_points','metadata_passed_points','canonical_chi2_verified_points','errors','warnings','ready_for_reprofile')},ensure_ascii=False))
        return 0 if errors==0 and self.point_rows else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=HERE/'upper_grid_manifest.csv')
    p.add_argument('--controls',type=Path,default=HERE/'control_points_manifest.csv')
    p.add_argument('--seed-dir',type=Path,default=HERE)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--verify-chi2',action='store_true')
    return Audit(p.parse_args(argv)).run()


if __name__=='__main__':raise SystemExit(main())
