"""Reproducible profiling of saved theories; never overwrites production fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize

from .inputs import PARAMETER_NAMES, load_context, load_manifest, load_snapshot, select_pilot
from .profiler import StableProfiler, ProfileNumericsError
from .residuals import FullResidual

PACKAGE = Path(__file__).resolve().parents[1]
SUPPORTED_FIT_SHA256 = '845ec6896c8901df8535d2bc8069344842235e1fc38ba4bca7fd5a59acfb30b6'
PILOT_KEYS = ['14.7142857143|-1.3750000000', '15.0000000000|-1.3750000000',
              '19.2857142857|-4.0000000000', '14.7142857143|-0.1000000000']
ANCHORS = ['16.0000000000|0.0000000000', '16.4285714286|-0.2000000000',
           '14.7142857143|-0.1000000000']
WITNESSES = {'14.7142857143|-1.3750000000': 192.053545084295,
             '15.0000000000|-1.3750000000': 193.904599510029}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate(ctx, theta, origin):
    """The scalar and saved vector always refer to exactly the same parameters."""
    theta = np.asarray(theta, float)
    bounds = np.asarray(ctx['bounds'], float)
    if theta.shape != (6,) or not np.isfinite(theta).all():
        raise ValueError('Nonfinite or malformed candidate')
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        raise ValueError('Candidate outside the saved nuisance domain')
    chi2 = float(ctx['fitter'].chi2(theta))
    if not np.isfinite(chi2) or chi2 >= 1e99:
        raise ValueError('Invalid canonical chi2')
    return {'theta': theta.tolist(), 'chi2': chi2, 'origin': str(origin)}


def best_of(*candidates):
    return min((c for c in candidates if c is not None), key=lambda c: c['chi2'])


def same_vector(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def neighbor_keys(snapshot, key):
    """Nearest stored points on each axis, used only as starts, never interpolation."""
    row = snapshot.loc[key]
    found = []
    for fixed, varying in [('log10m_acc', 'log10f_acc'), ('log10f_acc', 'log10m_acc')]:
        same = snapshot.loc[np.isclose(snapshot[fixed], row[fixed], rtol=0, atol=5e-10)]
        for sign in [-1, 1]:
            eligible = same.loc[sign * (same[varying] - row[varying]) > 5e-10]
            if not eligible.empty:
                closest = (eligible[varying] - row[varying]).abs().idxmin()
                found.append(str(closest))
    return list(dict.fromkeys(found))


class BudgetReached(RuntimeError):
    pass


class Trace:
    def __init__(self, ctx, budget):
        self.ctx = ctx
        self.budget = int(budget)
        self.calls = 0
        self.jac_calls = 0
        self.invalid = 0
        self.best = None

    def count(self, jac=False):
        if self.calls >= self.budget:
            raise BudgetReached(f'Budget {self.budget} reached')
        self.calls += 1
        self.jac_calls += int(jac)

    def retain(self, theta, origin='optimizer_visited'):
        c = candidate(self.ctx, theta, origin)
        self.best = c if self.best is None else best_of(self.best, c)
        return c


def optimize_attempt(ctx, profiler, start, *, method, budget, trial_id,
                     direction='none', stage='local', seed=None, start_label='own'):
    baseline = candidate(ctx, ctx['theta'], 'baseline')
    original_start = candidate(ctx, start, 'start')
    trace = Trace(ctx, budget)
    start_time = time.perf_counter()
    q_indices = np.array([1, 2, 3, 5])
    q0 = np.asarray(start, float)[q_indices]
    qbounds = np.asarray(profiler.bounds4, float)
    # Multiplication/division only; amplitudes are never mapped through a box midpoint.
    scales = np.maximum(qbounds[:, 1] - qbounds[:, 0], 1.0)
    ubounds = qbounds / scales[:, None]
    profiled_start = None
    if method != 'least_squares_full6':
        _, theta, _ = profiler.profile(q0)
        profiled_start = candidate(ctx, theta, 'profiled_start')

    def decode_q(u):
        q = np.asarray(u, float) * scales
        if np.any(q < qbounds[:,0]-1e-11) or np.any(q > qbounds[:,1]+1e-11):
            raise ValueError('Reduced coordinate outside bounds')
        return np.minimum(np.maximum(q,qbounds[:,0]),qbounds[:,1])

    def objective(u):
        trace.count()
        try:
            q = decode_q(u)
            _, theta, _ = profiler.profile(q)
            return trace.retain(theta)['chi2']
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, ProfileNumericsError):
            trace.invalid += 1
            return 1e100

    result = None
    final = None
    termination = 'completed'
    error = None
    try:
        if method == 'L-BFGS-B_profile4':
            result = minimize(objective, q0 / scales, method='L-BFGS-B',
                              bounds=ubounds, options={'maxfun': budget, 'maxiter': budget,
                              'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 40})
            _, theta, _ = profiler.profile(decode_q(result.x))
            final = candidate(ctx, theta, 'optimizer_final')
        elif method == 'DE_profile4':
            # The manual counter also covers any final partial generation.
            result = differential_evolution(objective, ubounds, seed=int(seed),
                popsize=12, maxiter=max(1, int(math.ceil(budget / 48))),
                tol=1e-7, atol=1e-6, polish=False, workers=1,
                updating='immediate', x0=q0 / scales)
            _, theta, _ = profiler.profile(decode_q(result.x))
            final = candidate(ctx, theta, 'optimizer_final')
        elif method == 'least_squares_full6':
            residuals = FullResidual(ctx)

            def fun(theta):
                trace.count()
                r = residuals.residual(theta)
                trace.retain(theta)
                return r

            def jac(theta):
                trace.count(jac=True)
                return residuals.jac(theta)

            result = least_squares(fun, np.asarray(start, float), jac=jac,
                bounds=np.asarray(ctx['bounds']).T, method='trf', loss='linear',
                x_scale=residuals.scales(start), ftol=1e-11, xtol=1e-11,
                gtol=1e-8, max_nfev=budget)
            final = candidate(ctx, result.x, 'optimizer_final')
        else:
            raise ValueError(f'Unknown method {method}')
        if not result.success:
            termination = 'not_converged'
    except BudgetReached as exc:
        termination = 'budget_exhausted'
        error = str(exc)
    except Exception as exc:
        termination = 'error'
        error = f'{type(exc).__name__}: {exc}'

    selected = best_of(baseline, original_start, profiled_start, trace.best, final)
    selected = candidate(ctx, selected['theta'], selected['origin'])
    row = dict(trial_id=trial_id, target=ctx['row']['coordinate_key'],
        stage=stage, direction=direction, method=method, seed=seed, start_label=start_label,
        chi2_old=baseline['chi2'], chi2_start=original_start['chi2'],
        chi2_profiled_start=None if profiled_start is None else profiled_start['chi2'],
        chi2_optimizer_final=None if final is None else final['chi2'],
        chi2_best_visited=None if trace.best is None else trace.best['chi2'],
        chi2_selected=selected['chi2'], gain=baseline['chi2']-selected['chi2'],
        selected_origin=selected['origin'], optimizer_success=bool(result.success) if result is not None else False,
        termination=termination, message=error or str(result.message), n_calls=trace.calls,
        n_jac_calls=trace.jac_calls, n_invalid=trace.invalid, budget=budget,
        elapsed_seconds=time.perf_counter()-start_time,
        has_new_vector=not same_vector(selected['theta'], baseline['theta']) and
                       not same_vector(selected['theta'], original_start['theta']),
        theta_old=baseline['theta'], theta_start=original_start['theta'],
        theta_profiled_start=None if profiled_start is None else profiled_start['theta'],
        theta_optimizer_final=None if final is None else final['theta'],
        theta_best_visited=None if trace.best is None else trace.best['theta'],
        theta_selected=selected['theta'])
    for name, value in zip(PARAMETER_NAMES, selected['theta']):
        row['parameter_' + name] = value
    return clean(row), selected


def paths_for_sweeps(manifest):
    """Cuts retain their real gaps; no uncomputed parameter point is inserted."""
    def select(mask, axis):
        return manifest.loc[mask].sort_values(axis).coordinate_key.tolist()
    m, f = manifest.log10m_acc, manifest.log10f_acc
    paths = []
    for fraction in [-0.3, 0.0]:
        paths.append(select(np.isclose(f, fraction) & (m >= 15.8) & (m < 17), 'log10m_acc'))
    for mass in [14.7142857143, 16.4285714286]:
        paths.append(select(np.isclose(m, mass, rtol=0, atol=5e-10) & (f >= -0.400001), 'log10f_acc'))
    paths.append(select(np.isclose(f, -1.375), 'log10m_acc'))
    for fraction in [-0.3, -0.2]:
        paths.append(select(np.isclose(f, fraction) & (m >= 17) & (m < 18), 'log10m_acc'))
    paths.append(select(f < -3, 'log10m_acc'))
    return [p for p in paths if p]


class Campaign:
    def __init__(self, project_root, snapshot_path, manifest_path, output, *,
                 local_budget=5000, global_budget=20000, seeds=(12345,23456),
                 data_dir=None, resume=False):
        self.root = Path(project_root).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.snapshot = load_snapshot(snapshot_path)
        frozen_snapshot = load_snapshot(PACKAGE/'baseline_selected_raw_points.csv')
        missing = frozen_snapshot.index.difference(self.snapshot.index)
        if len(missing):
            raise ValueError(f'A full snapshot is required for the reference and neighbor starts; '
                             f'{len(missing)} baseline coordinates are missing.')
        if not np.isfinite(self.snapshot.chi2_one_loop).all():
            raise ValueError('Snapshot contains nonfinite chi2 values.')
        self.manifest = load_manifest(manifest_path)
        self.points = select_pilot(self.snapshot, self.manifest)
        self.manifest_path = Path(manifest_path)
        self.local_budget, self.global_budget = int(local_budget), int(global_budget)
        self.seeds = tuple(int(s) for s in seeds)
        self.data_dir = data_dir
        self.contexts, self.profilers = {}, {}
        self.identity = dict(project_root=str(self.root), snapshot_sha256=file_hash(snapshot_path),
            manifest_sha256=file_hash(manifest_path), source_sha256=SUPPORTED_FIT_SHA256,
            package_sources={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            local_budget=self.local_budget, global_budget=self.global_budget, seeds=list(self.seeds),
            data_dir=None if data_dir is None else str(Path(data_dir).resolve()))
        state_path = self.output/'state.json'
        if state_path.exists():
            if not resume:
                raise ValueError('Output already contains state.json; use --resume or a new --out.')
            self.state = json.loads(state_path.read_text())
            if self.state['identity'] != self.identity:
                raise ValueError('Resume identity differs (code/data/budgets). Use a new output directory.')
        else:
            if resume:
                raise ValueError('Cannot resume: state.json does not exist.')
            self.state = {'identity':self.identity,'attempts':{},'best':{},'validation':{},
                          'directions':{},'pilot_passed':False}
        self.reference = float(self.snapshot.chi2_one_loop.min())
        self.save()

    def save(self):
        write_json(self.output/'state.json', self.state)

    def get_context(self, key):
        if key not in self.contexts:
            ctx = load_context(self.root, self.points.loc[key], self.data_dir)
            if ctx['source_sha256'] != SUPPORTED_FIT_SHA256:
                raise ValueError('Unsupported fit.py hash. The analytic profile must be audited for this version.')
            ctx['row'] = self.points.loc[key].to_dict()
            profiler = StableProfiler(ctx)
            validation = profiler.validate()
            residual_validation = FullResidual(ctx).validate(ctx['theta'])
            if not residual_validation.get('passed', False):
                self.state['validation'][key] = clean({'status':'failed',
                    'residual_checks':residual_validation})
                self.save()
                raise ValueError(f'Residual/Jacobian validation failed for {key}; see validation.json/state.json.')
            validated_metadata = clean({**ctx.get('validation_metadata', {}),'status':'validated',
                'chi2_saved':ctx['chi2_saved'],'chi2_recomputed':ctx['chi2_recomputed'],
                'profile_checks':validation,'residual_checks':residual_validation,
                'source_sha256':ctx['source_sha256'],
                'theory_digest':ctx['record']['theory_digest'],
                'bounds':ctx['bounds'], 'numerics':ctx.get('metadata_numerics', {})})
            old_metadata = self.state['validation'].get(key)
            if old_metadata:
                immutable=['fit_sha256','theory_sha256','covariance_sha256','source_sha256',
                           'data_z_sha256','data_k_velocity_sha256','data_p1d_sha256','bounds']
                if any(old_metadata.get(k)!=validated_metadata.get(k) for k in immutable):
                    raise ValueError(f'Inputs changed since checkpoint for {key}; use a new output directory.')
            old = candidate(ctx, ctx['theta'], 'baseline')
            previous = self.state['best'].get(key)
            if previous:
                previous = candidate(ctx, previous['theta'], previous['origin'])
            self.state['best'][key] = best_of(old, previous)
            self.contexts[key] = ctx
            self.profilers[key] = profiler
            self.state['validation'][key] = validated_metadata
            self.save()
        return self.contexts[key]

    def starts(self, key, extra=()):
        ctx = self.get_context(key)
        bank = [('own', candidate(ctx, ctx['theta'], 'baseline'))]
        for donor in neighbor_keys(self.snapshot, key):
            theta = [self.snapshot.loc[donor, 'parameter_'+n] for n in PARAMETER_NAMES]
            try:
                bank.append((donor, candidate(ctx, theta, 'neighbor_start')))
            except ValueError:
                continue
        for label, value in extra:
            try:
                bank.append((label, candidate(ctx, value['theta'], value.get('origin', 'start'))))
            except ValueError:
                continue
        unique = []
        for label,c in sorted(bank, key=lambda item:item[1]['chi2']):
            if not any(same_vector(c['theta'], old['theta']) for _,old in unique):
                unique.append((label,c))
        return unique

    def attempt(self, key, start, label, trial, method, *, direction='none',stage='local',seed=None):
        ctx = self.get_context(key)
        if trial in self.state['attempts']:
            stored = self.state['attempts'][trial]
            return candidate(ctx, stored['theta_selected'], stored['selected_origin'])
        print(f'{stage} {direction} {key}: {method}, start={label}', flush=True)
        row, selected = optimize_attempt(ctx,self.profilers[key],start['theta'],
            method=method,budget=self.global_budget if stage=='global' else self.local_budget,
            trial_id=trial,direction=direction,stage=stage,seed=seed,start_label=label)
        self.state['attempts'][trial] = row
        self.state['best'][key] = best_of(self.state['best'][key], selected)
        self.save()
        print(f"  chi2={selected['chi2']:.9f}; {row['termination']}; calls={row['n_calls']}; "
              f"origin={row['selected_origin']}; {row['elapsed_seconds']:.1f}s",flush=True)
        return selected

    def validate(self, keys=None):
        keys = list(self.points.index if keys is None else keys)
        keys = list(dict.fromkeys([*self.state['best'].keys(), *keys]))
        for key in keys:
            self.get_context(key)
        self.export()

    def run_pilot(self):
        keys = [k for k in PILOT_KEYS if k in self.points.index]
        if len(keys) != 4:
            raise ValueError('The pilot manifest must include all four pilot points.')
        self.validate(keys)
        for key in keys:
            bank = self.starts(key)
            for i,(label,start) in enumerate(bank[:2]):
                self.attempt(key,start,label,f'pilot/{key}/local/{i}', 'L-BFGS-B_profile4',stage='pilot')
            self.attempt(key,self.state['best'][key],'best_candidate',f'pilot/{key}/full6',
                         'least_squares_full6',stage='pilot')
            self.export()
        checks = {'all_four_validated':all(k in self.contexts for k in keys),
                  'known_witnesses_retained':all(self.state['best'][k]['chi2'] <= value+1e-3
                     for k,value in WITNESSES.items()),
                  'no_worse_than_baseline':all(self.state['best'][k]['chi2'] <=
                      self.contexts[k]['chi2_saved']+1e-8 for k in keys)}
        self.state['pilot_passed'] = all(checks.values())
        write_json(self.output/'pilot_gate.json', {'passed':self.state['pilot_passed'],
            'checks':checks,'note':'Correctness and recovery gate; not proof of optimizer/global convergence.',
            'identity':self.identity})
        self.save()
        self.export()
        if not self.state['pilot_passed']:
            raise RuntimeError('Pilot gate failed. Inspect pilot_gate.json before running the campaign.')

    def run_campaign(self):
        if not self.state.get('pilot_passed'):
            raise RuntimeError('Run --phase pilot first, then --phase campaign --resume with the same --out.')
        self.validate()
        # Independent anchors are shared by both directions and frozen before sweeps.
        for key in ANCHORS:
            start_label,start = self.starts(key)[0]
            for seed in self.seeds:
                self.attempt(key,start,start_label,f'global/{key}/{seed}','DE_profile4',stage='global',seed=seed)
            self.export()
        frozen_key = 'frozen_before_sweeps'
        if frozen_key not in self.state:
            self.state[frozen_key] = clean(self.state['best'])
            self.save()
        frozen = self.state[frozen_key]
        paths = paths_for_sweeps(self.manifest)
        for direction in ['forward','backward']:
            propagated = {}
            processed = set()
            for path in paths:
                previous = None
                sequence = path if direction=='forward' else list(reversed(path))
                for key in sequence:
                    if key in processed:
                        previous = propagated[key]
                        continue
                    extras = [('frozen_candidate', frozen[key])]
                    if previous is not None:
                        extras.append(('previous_'+direction,previous))
                    bank = self.starts(key,extras)
                    results = [frozen[key]]
                    # Preserve the continuation candidate even when its initial chi2 is worse.
                    continuation = [(label,c) for label,c in bank if label=='previous_'+direction]
                    selected_starts = bank[:1]
                    for item in continuation[:1]:
                        if not any(same_vector(item[1]['theta'],v[1]['theta']) for v in selected_starts):
                            selected_starts.append(item)
                    for item in bank:
                        if len(selected_starts)>=2:break
                        if not any(same_vector(item[1]['theta'],v[1]['theta']) for v in selected_starts):
                            selected_starts.append(item)
                    for i,(label,start) in enumerate(selected_starts):
                        results.append(self.attempt(key,start,label,f'{direction}/{key}/{i}',
                            'L-BFGS-B_profile4',direction=direction))
                    propagated[key] = best_of(*results)
                    previous = propagated[key]
                    processed.add(key)
                    self.state['directions'].setdefault(direction,{})[key] = propagated[key]
                    self.save()
                    self.export()
        for key in self.points.index:
            crosscheck_starts = self.state.setdefault('crosscheck_starts', {})
            if key not in crosscheck_starts:
                starts = [self.state['directions'][d][key] for d in ['forward','backward']]
                starts.append(self.state['best'][key])
                unique=[]
                for start in sorted(starts,key=lambda c:c['chi2']):
                    if not any(same_vector(start['theta'],old['theta']) for old in unique):unique.append(start)
                # Keep trial identities stable if the first attempt improves best
                # and the process is interrupted before the second attempt.
                crosscheck_starts[key] = clean(unique[:2])
                self.save()
            for i,start in enumerate(crosscheck_starts[key]):
                self.attempt(key,start,'direction_candidate',f'crosscheck/{key}/{i}',
                             'least_squares_full6',stage='crosscheck')
            self.export()
        self.state['campaign_completed'] = True
        self.save()
        self.export()

    def export(self):
        attempts = list(self.state['attempts'].values())
        flat=[]
        for row in attempts:
            flat.append({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        pd.DataFrame(flat).to_csv(self.output/'reprofile_attempts.csv',index=False)
        rows,predictions,by_redshift=[],[],[]
        for key,ctx in self.contexts.items():
            old=np.asarray(ctx['theta'],float)
            best=candidate(ctx,self.state['best'][key]['theta'],self.state['best'][key]['origin'])
            theta=np.asarray(best['theta'],float)
            point=ctx['row']
            relevant=[a for a in attempts if a['target']==key]
            completed=[a for a in relevant if a['termination']=='completed' and a['optimizer_success']]
            row=dict(target=key,log10m_acc=point['log10m_acc'],log10f_acc=point['log10f_acc'],
                chi2_old=ctx['chi2_saved'],chi2_best=best['chi2'],gain=ctx['chi2_saved']-best['chi2'],
                chi2_reference=self.reference,delta_chi2_old=ctx['chi2_saved']-self.reference,
                delta_chi2_best=best['chi2']-self.reference,selected_origin=best['origin'],
                validation_status='validated',optimization_status='completed_attempts' if completed else 'candidate_only',
                attempts=len(relevant),budget_exhausted=sum(a['termination']=='budget_exhausted' for a in relevant),
                optimizer_successes=len(completed),theory_digest=ctx['record']['theory_digest'])
            for i,name in enumerate(PARAMETER_NAMES):
                row.update({f'parameter_old_{name}':old[i],f'parameter_best_{name}':theta[i],
                            f'lower_{name}':ctx['bounds'][i,0],f'upper_{name}':ctx['bounds'][i,1]})
            for direction in ['forward','backward']:
                c=self.state['directions'].get(direction,{}).get(key)
                row['chi2_'+direction]=None if c is None else c['chi2']
            row['direction_spread']=None if row['chi2_forward'] is None or row['chi2_backward'] is None else abs(row['chi2_forward']-row['chi2_backward'])
            row['stability_assessment']='not_established'
            if row['direction_spread'] is not None:
                tolerance=0.003 if abs(row['delta_chi2_best']-5.991)<0.05 else 0.01
                row['agreement_tolerance']=tolerance
                # A preserved baseline is never treated as an optimizer convergence test.
                direction_finals=[]
                for direction in ['forward','backward']:
                    vals=[a['chi2_optimizer_final'] for a in relevant if a['direction']==direction and a['optimizer_success'] and a['chi2_optimizer_final'] is not None]
                    direction_finals.append(min(vals) if vals else np.nan)
                other=[a['chi2_optimizer_final'] for a in relevant if a['method']=='least_squares_full6' and a['optimizer_success'] and a['chi2_optimizer_final'] is not None]
                all_final=direction_finals+([min(other)] if other else [np.nan])
                if np.isfinite(all_final).all() and max(abs(v-best['chi2']) for v in all_final)<=tolerance:
                    row['stability_assessment']='agreement_observed_not_global_proof'
                else:row['stability_assessment']='further_search_needed'
            rows.append(row)
            dataset=ctx['dataset']; fitter=ctx['fitter']
            pred_old=np.asarray(fitter.model(old));pred_new=np.asarray(fitter.model(theta))
            sigma=np.sqrt(np.diag(fitter.covariance))
            for i in range(len(pred_old)):
                predictions.append(dict(target=key,z=dataset.z[i],k_velocity=dataset.k_velocity[i],
                    data=dataset.p1d[i],sigma=sigma[i],prediction_old=pred_old[i],prediction_best=pred_new[i]))
            oldz={float(r['z']):r for r in fitter.chi2_by_redshift(old)}
            for zrow in fitter.chi2_by_redshift(theta):
                by_redshift.append(dict(target=key,z=zrow['z'],n_data=zrow['n_data'],
                    chi2_old=oldz[float(zrow['z'])]['chi2'],chi2_best=zrow['chi2']))
        pd.DataFrame(rows).to_csv(self.output/'reprofile_best_points.csv',index=False)
        pd.DataFrame(predictions).to_csv(self.output/'reprofile_p1d.csv',index=False)
        pd.DataFrame(by_redshift).to_csv(self.output/'reprofile_chi2_by_redshift.csv',index=False)
        write_json(self.output/'validation.json',self.state['validation'])
        write_json(self.output/'campaign_metadata.json',dict(identity=self.identity,
            chi2_reference=self.reference,pilot_passed=self.state['pilot_passed'],
            campaign_completed=self.state.get('campaign_completed',False),
            available_targets=list(self.contexts),n_targets_planned=len(self.points),
            note='All best points are feasible candidates. Convergence/global optimality is not inferred from retention.'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path)
    parser.add_argument('--snapshot',type=Path,default=PACKAGE/'baseline_selected_raw_points.csv')
    parser.add_argument('--manifest',type=Path,default=PACKAGE/'pilot_points.csv')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=['validate','pilot','campaign','report'],default='pilot')
    parser.add_argument('--local-budget',type=int,default=5000)
    parser.add_argument('--global-budget',type=int,default=20000)
    parser.add_argument('--seeds',type=int,nargs=2,default=[12345,23456])
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.phase=='report':
        from .report import make_report
        make_report(args.out,args.manifest)
        return 0
    if args.project_root is None:
        parser.error('--project-root is required for calculations')
    if min(args.local_budget,args.global_budget)<100:
        parser.error('Budgets must be at least 100 evaluations')
    runner=Campaign(args.project_root,args.snapshot,args.manifest,args.out,
        local_budget=args.local_budget,global_budget=args.global_budget,seeds=args.seeds,
        data_dir=args.data_dir,resume=args.resume)
    try:
        if args.phase=='validate':runner.validate()
        elif args.phase=='pilot':runner.run_pilot()
        else:runner.run_campaign()
    except Exception as exc:
        runner.save()
        runner.export()
        write_json(args.out/'last_error.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
    from .report import make_report
    make_report(args.out,args.manifest)
    print(f'Outputs: {args.out.resolve()}',flush=True)
    return 0
