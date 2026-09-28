"""Stage A coordinator: immutable rounds, private point files, checked resume."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

from accdm_reprofile.inputs import load_snapshot, load_manifest, resolve_path, PARAMETER_NAMES
from accdm_reprofile.campaign import clean, write_json, SUPPORTED_FIT_SHA256
from .bank import neighbors, seed_bank, global_representatives, proposals_for, pilot_keys
from .worker import run_point

PACKAGE = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_inputs(args):
    root, data = args.project_root.resolve(), args.package_data.resolve()
    audit = args.preflight.resolve() if args.preflight else data / 'audited_inputs'
    summary = read_json(audit / 'preflight_summary.json')
    if not summary.get('ready_for_reprofile') or summary.get('errors') != 0:
        raise ValueError('Preflight is not ready_for_reprofile, or has errors.')
    if not summary.get('canonical_chi2_requested'):
        raise ValueError('Canonical chi2 audit is required.')
    if sha(root / 'lyalpha_pt/fit.py') != SUPPORTED_FIT_SHA256:
        raise ValueError('Unsupported canonical fit.py. Do not bypass the code check.')
    snapshot = load_snapshot(data / 'baseline_full_snapshot.csv')
    upper = load_manifest(data / 'upper_grid_manifest.csv')
    controls = load_manifest(data / 'control_points_manifest.csv')
    keys = list(upper.index) + list(controls.index)
    if len(set(keys)) != len(keys) or not set(keys).issubset(snapshot.index):
        raise ValueError('Duplicate targets or targets absent from snapshot.')
    audited = pd.read_csv(audit / 'preflight_points.csv').set_index('coordinate_key', drop=False)
    if audited.index.duplicated().any() or not set(keys).issubset(audited.index):
        raise ValueError('Preflight does not cover each requested point exactly once.')
    if not (audited.loc[keys, 'canonical_chi2_status'] == 'verified').all():
        raise ValueError('Not all selected points have canonical chi2 verification.')
    if not (audited.loc[keys, 'metadata_status'] == 'passed').all():
        raise ValueError('Not all selected points passed metadata verification.')
    print(f'Checking current input hashes for {len(keys)} points ...', flush=True)
    checked = {}
    for key in keys:
        row, previous = snapshot.loc[key], audited.loc[key]
        if str(row.theory_digest) != str(previous.bundle_digest):
            raise ValueError(f'Snapshot/theory digest mismatch: {key}')
        for col, hcol in [('fit_path', 'fit_sha256'), ('theory_path', 'theory_sha256')]:
            path = resolve_path(row[col], root)
            if path != resolve_path(previous[col], root):
                raise ValueError(f'Audited path differs for {key}: {col}')
            if str(path) not in checked:
                checked[str(path)] = sha(path)
            digest = checked[str(path)]
            if digest != str(previous[hcol]):
                raise ValueError(f'Input changed after preflight: {path}')
    for old_dir, record in summary.get('data_directories', {}).items():
        location = args.data_dir.resolve() if args.data_dir else resolve_path(old_dir, root)
        for name, meta in record['files'].items():
            path = location / name
            if sha(path) != meta['sha256']:
                raise ValueError(f'DR12 changed after preflight: {path}')
    if not summary.get('data_directories'):
        raise ValueError('Missing audited DR12 hashes.')
    input_files = {str(p.relative_to(data)): sha(p) for p in [
        data/'baseline_full_snapshot.csv', data/'upper_grid_manifest.csv',
        data/'control_points_manifest.csv', data/'seed_best_points.csv', data/'seed_attempts.csv']}
    package_code = {str(p.relative_to(PACKAGE)): sha(p)
        for folder in ('accdm_upper_grid', 'accdm_reprofile')
        for p in sorted((PACKAGE / folder).glob('*.py'))}
    project_code = {str(p.relative_to(root)): sha(p) for p in sorted((root/'lyalpha_pt').glob('*.py'))}
    identity = dict(project_root=str(root), data_dir=str(args.data_dir.resolve()) if args.data_dir else None,
        input_files=input_files, package_code=package_code, project_code=project_code,
        audit_summary=sha(audit/'preflight_summary.json'), audit_points=sha(audit/'preflight_points.csv'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    return snapshot, upper, controls, audited, identity


def branch_changed(old, new):
    """Conservative shape test; predictions are additionally deduplicated in worker."""
    indices = [1, 2, 3, 5]
    scales = np.array([30., 16., 35., 2000.])
    return any(not any(np.linalg.norm((np.asarray(c['theta'])[indices] -
        np.asarray(p['theta'])[indices]) / scales) <= 1e-4 for p in old) for c in new)


def export_round(out, results, snapshot, keys, bank, round_index, round_stats, stable_count, phase):
    reference_old = float(snapshot.chi2_one_loop.min())
    all_chi = snapshot.chi2_one_loop.to_dict()
    all_chi.update({key: result['best']['chi2'] for key, result in results.items()})
    reference_best = float(min(all_chi.values()))
    rows, attempt_rows = [], []
    for key in keys:
        result, old = results[key], snapshot.loc[key]
        row = dict(target=key, log10m_acc=float(old.log10m_acc), log10f_acc=float(old.log10f_acc),
            chi2_old=float(old.chi2_one_loop), chi2_best=result['best']['chi2'],
            gain=float(old.chi2_one_loop)-result['best']['chi2'],
            delta_chi2_fixed_reference=result['best']['chi2']-reference_old,
            delta_chi2_updated_reference=result['best']['chi2']-reference_best,
            solver_verified=result['solver_verified'], bank_size=len(result['bank']),
            nearest_boundary_fraction=result['nearest_boundary_fraction'],
            theory_digest=old.theory_digest, round_index=round_index)
        for name, value in zip(PARAMETER_NAMES, result['best']['theta']):
            row['parameter_best_'+name] = value
        rows.append(row)
        attempt_rows.extend(result['attempts'])
    frame = pd.DataFrame(rows)
    csv_temp = out / 'reprofile_best_points.csv.tmp'
    frame.to_csv(csv_temp, index=False)
    csv_temp.replace(out / 'reprofile_best_points.csv')
    pd.DataFrame(attempt_rows).to_csv(out / 'latest_round_attempts.csv', index=False)
    write_json(out / 'candidate_bank.json', {k: bank[k] for k in keys})
    summary = dict(phase=phase, stage='A_original_bounds_continuation', n_points=len(keys),
        last_completed_round=round_index, stable_rounds=stable_count,
        propagation_stable=stable_count>=2, global_minimum_certified=False,
        exhaustive_global_search_performed=False, bounds_changed=False,
        old_reference_chi2=reference_old, updated_reference_chi2=reference_best,
        reference_uses_untouched_baselines_outside_targets=True,
        n_improved_gt_0p01=int((frame.gain>0.01).sum()), maximum_gain=float(frame.gain.max()),
        n_solver_unverified=int((~frame.solver_verified.astype(bool)).sum()),
        n_near_bound_5pct=int((frame.nearest_boundary_fraction<0.05).sum()),
        last_round=round_stats,
        note='Local continuation only. Review unresolved points and run global/bounds/theory tests separately.')
    write_json(out / 'summary.json', summary)
    return summary


def run(args):
    started = time.perf_counter()
    snapshot, upper, controls, audited, inputs_identity = load_inputs(args)
    config = dict(local_budget=args.local_budget, escalated_budget=args.escalated_budget,
                  max_starts=4, bank_size=6)
    bank = seed_bank(snapshot, args.package_data/'seed_best_points.csv', args.package_data/'seed_attempts.csv')
    best_keys = pd.read_csv(args.package_data/'seed_best_points.csv')
    keys = (pilot_keys(snapshot, list(upper.index), list(controls.index), best_keys, n=32)
            if args.phase=='pilot' else list(upper.index)+list(controls.index))
    seed_import = None
    if args.seed_run:
        seed_meta = read_json(args.seed_run/'campaign.json')
        if seed_meta['inputs_fingerprint'] != signature(inputs_identity):
            raise ValueError('Seed-run has different inputs, code or environment.')
        seed_file = args.seed_run/'candidate_bank.json'
        imported = read_json(seed_file)
        for key, candidates in imported.items():
            if key not in bank:
                raise ValueError(f'Unknown seed-run key: {key}')
            bank[key].extend(candidates)
        seed_import = dict(path=str(args.seed_run.resolve()), sha256=sha(seed_file))
    identity = dict(inputs_fingerprint=signature(inputs_identity), inputs=inputs_identity,
        selected_keys=keys, config=config, phase=args.phase, seed_import=seed_import,
        stage='A_original_bounds_continuation_v1')
    fingerprint = signature(identity)
    out = args.out.resolve()
    # A fresh dedicated output directory is required; originals are never overwritten.
    if out == args.project_root.resolve() or out == args.package_data.resolve():
        raise ValueError('Choose a dedicated new output directory.')
    meta_path = out/'campaign.json'
    if out.exists() and not meta_path.exists():
        unexpected = [p.name for p in out.iterdir() if p.name != 'campaign.lock']
        if unexpected:
            raise ValueError(f'Output is not empty: {unexpected[:5]}')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'campaign.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another runner is already using this output directory.')
    if meta_path.exists():
        if not args.resume:
            raise ValueError('Output already has a campaign. Use --resume or a new --out.')
        if read_json(meta_path).get('fingerprint') != fingerprint:
            raise ValueError('Resume identity mismatch. Use the original inputs/options or a new output.')
    else:
        unexpected = [p.name for p in out.iterdir() if p.name != 'campaign.lock']
        if unexpected:
            raise ValueError(f'Output is not empty: {unexpected[:5]}')
        write_json(meta_path, dict(fingerprint=fingerprint, **identity))
    snapshot.loc[keys].to_csv(out/'selected_points.csv', index=False)
    neighbor_map = neighbors(snapshot)
    stable_count = 0
    summary = None
    # Spawn prevents worker imports/caches from leaking across project roots.
    context = multiprocessing.get_context('spawn')
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=context) if args.workers>1 else None
    try:
        for round_index in range(1, args.max_rounds+1):
            round_start = time.perf_counter()
            globals_bank = global_representatives({k: bank[k] for k in keys}, limit=8)
            directory = out/'rounds'/f'{round_index:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            results, pending = {}, []
            for key in keys:
                proposals = proposals_for(key, bank, neighbor_map, globals_bank)
                task_data = dict(key=key, row=snapshot.loc[key].to_dict(),
                    project_root=str(args.project_root.resolve()),
                    data_dir=str(args.data_dir.resolve()) if args.data_dir else None,
                    proposals=proposals, previous_bank=bank[key], round_index=round_index,
                    config=config, expected={name: str(audited.loc[key,name])
                        for name in ['fit_sha256','theory_sha256']})
                task_id = signature(dict(campaign=fingerprint, **task_data))
                path = directory/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.json')
                task = dict(task_data, task_id=task_id, out_path=str(path))
                if path.exists():
                    result = read_json(path)
                    if result.get('task_id') != task_id or result.get('key') != key:
                        raise ValueError(f'Checkpoint identity mismatch: {path}')
                    results[key] = result
                else:
                    pending.append(task)
            print(f'Round {round_index}: {len(keys)} points, {len(results)} cached, '
                  f'{len(pending)} to fit, workers={args.workers}', flush=True)
            failures = []
            if pool:
                futures = {pool.submit(run_point, task): task['key'] for task in pending}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        results[key] = future.result()
                    except Exception as exc:
                        failures.append(dict(key=key, error=f'{type(exc).__name__}: {exc}'))
                    if len(results)%16==0 or len(results)+len(failures)==len(keys):
                        print(f'  completed={len(results)}/{len(keys)}, errors={len(failures)}', flush=True)
            else:
                for task in pending:
                    try:
                        results[task['key']] = run_point(task)
                    except Exception as exc:
                        failures.append(dict(key=task['key'], error=f'{type(exc).__name__}: {exc}'))
                    print(f'  completed={len(results)}/{len(keys)}, errors={len(failures)}', flush=True)
            if failures:
                write_json(directory/'failures.json', failures)
                raise RuntimeError(f'{len(failures)} point failures; see {directory}/failures.json. '
                                   'Completed points are reusable with --resume.')
            failure_log = directory/'failures.json'
            if failure_log.exists():
                failure_log.unlink()
            reference = min([float(snapshot.chi2_one_loop.min())]+
                            [float(v['best']['chi2']) for v in results.values()])
            active_gain, changed = 0, 0
            gains = []
            for key in keys:
                result = results[key]
                prior = min(float(c['chi2']) for c in bank[key])
                gain = prior-float(result['best']['chi2'])
                gains.append(gain)
                if gain < -1e-6:
                    raise RuntimeError(f'Best candidate became worse at {key}: {gain}')
                near = abs(result['best']['chi2']-reference-5.991)<=1
                active_gain += gain > (0.003 if near else 0.01)
                changed += branch_changed(bank[key], result['bank'])
            verified = all(result['solver_verified'] for result in results.values())
            stable_count = stable_count+1 if active_gain==0 and changed==0 and verified else 0
            for key in keys:
                bank[key] = results[key]['bank']
            stats = dict(max_gain=float(max(gains)), n_significant_improvements=int(active_gain),
                n_changed_banks=int(changed), all_solver_verified=verified,
                wall_seconds=time.perf_counter()-round_start)
            write_json(directory/'round_summary.json', stats)
            summary = export_round(out, results, snapshot, keys, bank, round_index,
                                   stats, stable_count, args.phase)
            print(f'Round {round_index} done: max gain={max(gains):.6g}, '
                  f'changed banks={changed}, stable rounds={stable_count}', flush=True)
            if stable_count>=2:
                break
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        lock.close()
    summary['invocation_wall_seconds'] = time.perf_counter()-started
    summary['execution_status'] = 'finished'
    summary['scientific_status'] = ('locally_stable_requires_global_checks' if stable_count>=2
                                    else 'round_limit_reached_unresolved')
    write_json(out/'summary.json', summary)
    print(f'Outputs: {out}\nScientific status: {summary["scientific_status"]}', flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--package-data',type=Path,default=PACKAGE)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','full'],default='full')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--max-rounds',type=int,default=12)
    p.add_argument('--local-budget',type=int,default=2000)
    p.add_argument('--escalated-budget',type=int,default=5000)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--seed-run',type=Path)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if min(args.workers,args.max_rounds,args.local_budget,args.escalated_budget)<1:
        p.error('Workers, rounds and budgets must be positive.')
    if args.escalated_budget<args.local_budget:
        p.error('Escalated budget must be at least the local budget.')
    try:
        return run(args)
    except Exception as exc:
        print(f'BLOCKED: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
