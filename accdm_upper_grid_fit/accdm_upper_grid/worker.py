"""One Stage A continuation task on unchanged saved theory and nuisance bounds.

Local solver checks and bank stability are not a certificate of a global minimum.
Every stored candidate is evaluated on this target's canonical likelihood.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from accdm_reprofile.inputs import load_context, resolve_path, sha256_file, PARAMETER_NAMES
from accdm_reprofile.profiler import StableProfiler, ProfileNumericsError
from accdm_reprofile.residuals import FullResidual
from accdm_reprofile.campaign import optimize_attempt, candidate, clean, SUPPORTED_FIT_SHA256

sys.dont_write_bytecode = True
Q = np.array([1, 2, 3, 5])


def _atomic_json(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(clean(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


class _Diversity:
    """Deduplicate only when BOTH four-shape and whitened P1D agree.

    No absolute comparison of tiny alpha_ct is used. Prediction distance is
    RMS in units of the unchanged observational covariance.
    """
    def __init__(self, ctx, profiler):
        self.ctx, self.profiler = ctx, profiler
        self.width = np.diff(np.asarray(profiler.bounds4), axis=1).ravel()
        self.cache = {}

    def features(self, member):
        key = tuple(member['theta'])
        if key not in self.cache:
            theta = np.asarray(key, float)
            prediction = self.profiler.whiten(self.ctx['fitter'].model(theta))
            if not np.isfinite(prediction).all(): raise ValueError('Nonfinite bank prediction')
            self.cache[key] = (theta[Q]/self.width, prediction)
        return self.cache[key]

    def distance(self, a, b):
        qa, pa = self.features(a); qb, pb = self.features(b)
        return max(float(np.max(np.abs(qa-qb)))/1.e-3,
                   float(np.sqrt(np.mean((pa-pb)**2)))/1.e-3)

    def distinct(self, items):
        kept = []
        for item in sorted(items, key=lambda x: x['chi2']):
            if not any(self.distance(item, previous) <= 1. for previous in kept):
                kept.append(item)
        return kept

    def choose(self, items, limit, first_best=2):
        remaining = self.distinct(items)
        selected = remaining[:min(first_best, limit)]; remaining = remaining[len(selected):]
        while remaining and len(selected) < limit:
            index = max(range(len(remaining)), key=lambda i:
                (min(self.distance(remaining[i], old) for old in selected), -remaining[i]['chi2']))
            selected.append(remaining.pop(index))
        return selected


def run_point(task: dict) -> dict:
    """Evaluate proposals, continue several branches and atomically save one task.

    Integrity/validation exceptions propagate before an output is written.
    Individual optimizer failures are recorded and retried once; retention of
    an earlier candidate never changes a solver's reported termination status.
    """
    started = time.perf_counter()
    key = str(task['key']); row = dict(task['row'])
    if row.get('coordinate_key') != key: raise ValueError('Task key differs from row identity')
    root = Path(task['project_root']).expanduser().resolve()
    expected = task['expected']
    if sha256_file(root/'lyalpha_pt/fit.py') != SUPPORTED_FIT_SHA256:
        raise ValueError('Unsupported canonical fit.py; analytic core cannot be used')
    input_paths = {}
    for field, hash_name in [('fit_path','fit_sha256'), ('theory_path','theory_sha256')]:
        path = resolve_path(row.get(field), root); input_paths[field] = path
        if not expected.get(hash_name) or sha256_file(path) != expected[hash_name]:
            raise ValueError(f'{hash_name} differs from frozen task identity')
    out = Path(task['out_path']).expanduser().resolve()
    if out in input_paths.values() or root/'lyalpha_pt' in out.parents:
        raise ValueError('Output path would overwrite scientific input/source')
    if out.exists():
        existing = json.loads(out.read_text(encoding='utf-8'))
        if existing.get('task_id') != task['task_id']:
            raise ValueError('Output belongs to another task identity')
    ctx = load_context(root, row, task.get('data_dir'))
    if ctx['source_sha256'] != SUPPORTED_FIT_SHA256:
        raise ValueError('Canonical fit source changed during context construction')
    metadata = ctx['validation_metadata']
    for name in ('fit_sha256','theory_sha256'):
        if metadata.get(name) != expected[name]: raise ValueError(f'{name} changed during context construction')
    profiler = StableProfiler(ctx)
    profile_check = profiler.validate()
    residual_check = FullResidual(ctx).validate(ctx['theta'])
    if not profile_check.get('validated') or not residual_check.get('passed'):
        raise ValueError('Profile/full-residual numerical identity check failed')
    config = dict(task.get('config') or {})
    local_budget = int(config.get('local_budget',2000))
    escalated_budget = int(config.get('escalated_budget',5000))
    max_starts = int(config.get('max_starts',4)); bank_size = int(config.get('bank_size',6))
    if local_budget < 1 or escalated_budget < local_budget or not 1 <= max_starts <= 4 or not 1 <= bank_size <= 6:
        raise ValueError('Invalid Stage A budgets/bank limits')
    diversity = _Diversity(ctx, profiler)
    baseline = candidate(ctx, ctx['theta'], 'original_baseline')
    pool = [baseline]; previous = []
    for index, item in enumerate(task.get('previous_bank') or []):
        old = candidate(ctx, item['theta'], item.get('origin',f'previous_bank/{index}'))
        if not np.isfinite(float(item['chi2'])) or not np.isclose(old['chi2'],float(item['chi2']),atol=1e-4,rtol=1e-7):
            raise ValueError('Previous-bank candidate chi2 does not reproduce on its exact vector')
        previous.append(old); pool.append(old)
    previous_best = min(previous,key=lambda x:x['chi2']) if previous else None
    proposals = [dict(theta=baseline['theta'],origin='own_baseline',source_key=key)]
    proposals.extend(dict(theta=x['theta'],origin='previous/'+x['origin'],source_key=key) for x in previous)
    proposals.extend(task.get('proposals') or [])
    proposal_results = []; profiled_starts = []
    for index, item in enumerate(proposals):
        origin = str(item.get('origin',f'proposal/{index}'))
        report = dict(proposal_index=index,origin=origin,source_key=item.get('source_key'),
                      theta_input=clean(item.get('theta')),status='rejected')
        try:
            theta = np.asarray(item['theta'],float)
            if theta.shape != (6,) or not np.isfinite(theta[Q]).all():
                raise ValueError('Proposal must contain six entries and four finite shape parameters')
            q = theta[Q]
            if np.any(q < profiler.bounds4[:,0]) or np.any(q > profiler.bounds4[:,1]):
                raise ValueError('Proposal shape outside target original bounds; not clipped')
            amplitudes_inside = bool(np.isfinite(theta[[0,4]]).all() and
                np.all(theta[[0,4]] >= ctx['bounds'][[0,4],0]) and
                np.all(theta[[0,4]] <= ctx['bounds'][[0,4],1]))
            chi2, profiled_theta, diagnostics = profiler.profile(q)
            entry = candidate(ctx,profiled_theta,'profiled/'+origin)
            if not np.isclose(entry['chi2'],chi2,atol=1e-7,rtol=1e-11):
                raise ValueError('Profile result chi2/vector mismatch')
            report.update(status='profiled',q=q.tolist(),chi2_profiled=entry['chi2'],
                theta_profiled=entry['theta'],donor_amplitudes_inside_target_bounds=amplitudes_inside,
                amplitude_diagnostics=diagnostics)
            profiled_starts.append(entry); pool.append(entry)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, ProfileNumericsError) as error:
            report['message'] = f'{type(error).__name__}: {error}'
        proposal_results.append(clean(report))
    if not profiled_starts: raise RuntimeError('No proposal, including own baseline, could be profiled')
    starts = diversity.choose(profiled_starts,max_starts,first_best=2)
    attempts = []

    def attempt(start, method, label, budget, retry=False):
        trial_id = f"{task['task_id']}/{label}" + ('/retry' if retry else '')
        record, selected = optimize_attempt(ctx,profiler,start['theta'],method=method,budget=budget,
            trial_id=trial_id,stage='stage_a_full6' if method=='least_squares_full6' else 'stage_a_local',
            direction=f"round_{int(task['round_index'])}",start_label=start['origin'])
        record['retry'] = retry
        attempts.append(record)
        # A discarded optimizer endpoint may represent another useful branch.
        for theta_name,chi_name in [('theta_start','chi2_start'),('theta_profiled_start','chi2_profiled_start'),
                ('theta_optimizer_final','chi2_optimizer_final'),('theta_best_visited','chi2_best_visited'),
                ('theta_selected','chi2_selected')]:
            if record.get(theta_name) is not None:
                entry = candidate(ctx,record[theta_name],trial_id+'/'+theta_name.removeprefix('theta_'))
                if not np.isclose(entry['chi2'],float(record[chi_name]),atol=1e-7,rtol=1e-11):
                    raise RuntimeError('Optimizer log chi2/vector identity mismatch')
                pool.append(entry)
        if not retry and (not record['optimizer_success'] or record['termination']!='completed'):
            retry_start = candidate(ctx,selected['theta'],trial_id+'/retained_for_retry')
            attempt(retry_start,method,label,escalated_budget,retry=True)

    for index,start in enumerate(starts):
        attempt(start,'L-BFGS-B_profile4',f'local/{index}',local_budget)
    full_starts = diversity.choose(pool,2,first_best=2)
    for index,start in enumerate(full_starts):
        attempt(start,'least_squares_full6',f'full6/{index}',local_budget)
    best = min(pool,key=lambda x:x['chi2'])
    bank = diversity.choose(pool,bank_size,first_best=2)
    # Every selected vector is re-evaluated exactly, including the final best.
    best = candidate(ctx,best['theta'],best['origin'])
    bank = [candidate(ctx,x['theta'],x['origin']) for x in bank]
    if best['chi2'] > baseline['chi2'] or (previous_best and best['chi2'] > previous_best['chi2']):
        raise RuntimeError('Retention invariant failed')
    verified_endpoints = [x for x in attempts if x['method']=='least_squares_full6'
        and x['optimizer_success'] and x['termination']=='completed'
        and x['chi2_optimizer_final'] is not None
        and abs(float(x['chi2_optimizer_final'])-best['chi2'])<=.003]
    bounds=np.asarray(ctx['bounds']); theta=np.asarray(best['theta'])
    boundary=np.minimum(theta-bounds[:,0],bounds[:,1]-theta)/(bounds[:,1]-bounds[:,0])
    payload = dict(task_id=task['task_id'],key=key,round_index=int(task['round_index']),
        best=best,bank=bank,attempts=attempts,proposal_results=proposal_results,
        retained_baseline=baseline,retained_previous_best=previous_best,
        validation=clean({**metadata,'profile_checks':profile_check,'residual_checks':residual_check,
                          'bounds':bounds,'source_sha256':ctx['source_sha256']}),
        nearest_boundary_fraction=float(np.min(boundary)),
        boundary_fractions=dict(zip(PARAMETER_NAMES,boundary.tolist())),
        solver_verified=bool(verified_endpoints),
        solver_verification_trial_ids=[x['trial_id'] for x in verified_endpoints],
        unresolved_attempts=sum(not x['optimizer_success'] or x['termination']!='completed' for x in attempts),
        wall_seconds=time.perf_counter()-started,
        status='completed' if verified_endpoints else 'unresolved',
        search_scope='stage_a_local_multibranch_continuation_original_bounds',
        global_minimum_certified=False,
        diversity_criteria={'q_relative_box_tolerance':1e-3,'whitened_prediction_rms_tolerance':1e-3})
    payload=clean(payload)
    _atomic_json(out,payload)
    return payload
