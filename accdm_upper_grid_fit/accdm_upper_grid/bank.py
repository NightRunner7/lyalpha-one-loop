"""Deterministic candidate-bank helpers, with no production-project imports.

Recorded chi2 belongs only to source_key and is not a target score. Every
returned candidate must be re-evaluated/profiled by its target worker.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd

NAMES = ("log_alpha_F", "beta_F", "alpha_bias", "beta_bias", "alpha_ct", "beta_ct")
Q_INDICES = (1, 2, 3, 5)
Q_SCALES = np.array([30., 16., 35., 2000.])
ANCHORS = ("16.0000000000|0.0000000000", "16.4285714286|-0.2000000000",
           "14.7142857143|-0.1000000000")


def _frame(snapshot):
    if not isinstance(snapshot, pd.DataFrame):
        raise TypeError("snapshot must be a DataFrame indexed by coordinate_key")
    if snapshot.index.has_duplicates:
        raise ValueError("Duplicate snapshot coordinate keys")
    if "coordinate_key" in snapshot and any(str(k) != str(v)
            for k, v in zip(snapshot.index, snapshot.coordinate_key)):
        raise ValueError("Snapshot index and coordinate_key column disagree")
    return snapshot


def _coords(snapshot):
    result = {}
    for key, row in _frame(snapshot).iterrows():
        m, f = float(row["log10m_acc"]), float(row["log10f_acc"])
        if not np.isfinite([m, f]).all():
            raise ValueError(f"Nonfinite coordinates: {key}")
        canonical = "|".join("0.0000000000" if abs(v) < .5e-10 else f"{v:.10f}"
                             for v in (m, f))
        if str(key) != canonical:
            raise ValueError(f"Coordinate identity mismatch: {key}")
        result[str(key)] = (round(m, 10), round(f, 10))
    return result


def neighbors(snapshot):
    """Nearest stored neighbour in each native-axis direction, including gaps.

    List order is lower mass, higher mass, lower fraction, higher fraction.
    No interpolation or synthetic grid point is introduced.
    """
    coordinates = _coords(snapshot)
    by_mass, by_fraction = defaultdict(list), defaultdict(list)
    slots = {key: [None] * 4 for key in coordinates}
    for key, (m, f) in coordinates.items():
        by_mass[m].append((f, key))
        by_fraction[f].append((m, key))
    for groups, offset in ((by_fraction, 0), (by_mass, 2)):
        for values in groups.values():
            ordered = sorted(values)
            for i, (_, key) in enumerate(ordered):
                if i:
                    slots[key][offset] = ordered[i - 1][1]
                if i + 1 < len(ordered):
                    slots[key][offset + 1] = ordered[i + 1][1]
    return {key: [v for v in slots[key] if v is not None] for key in sorted(slots)}


def _table(value):
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return pd.read_csv(Path(value), dtype={"target": str, "trial_id": str})


def _number(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {name}")
    return value


def _theta(value):
    if isinstance(value, str):
        value = json.loads(value)
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("Expected six finite nuisance parameters")
    return value


def _identity(row, source, key):
    if not np.isclose(_number(row["chi2_old"], "chi2_old"),
                      float(source["chi2_one_loop"]), atol=1e-4, rtol=1e-7):
        raise ValueError(f"Seed chi2_old differs from baseline: {key}")
    if "theory_digest" in row and pd.notna(row["theory_digest"]):
        if str(row["theory_digest"]) != str(source["theory_digest"]):
            raise ValueError(f"Seed theory digest mismatch: {key}")
    for name in NAMES:
        for side in ("lower", "upper"):
            field = side + "_" + name
            if field in row and not np.isclose(_number(row[field], field),
                    float(source[field]), atol=1e-12, rtol=1e-12):
                raise ValueError(f"Seed bounds differ: {key} {field}")
        field = "parameter_old_" + name
        if field in row and not np.isclose(_number(row[field], field),
                float(source["parameter_" + name]), atol=0., rtol=1e-9):
            raise ValueError(f"Seed old parameter differs: {key} {name}")


def _entry(source, key, theta, chi2, origin):
    theta = _theta(theta)
    bounds = np.array([[source["lower_" + n], source["upper_" + n]] for n in NAMES], float)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(f"Invalid baseline bounds: {key}")
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        raise ValueError(f"Candidate outside source bounds: {key} {origin}")
    return {"theta": theta.tolist(), "chi2": _number(chi2, "candidate chi2"),
            "origin": str(origin), "source_key": str(key), "verified": False,
            "score_scope": "source_key_only_unverified",
            "log10m_acc": float(source["log10m_acc"]),
            "log10f_acc": float(source["log10f_acc"])}


def seed_bank(snapshot, best_csv=None, attempts_csv=None):
    """Collect old fits and all selected/final seed candidates without refits.

    Checks identity, recorded old chi2, exact saved source bounds and optional
    digests. This metadata check does not establish canonical chi2 agreement.
    Worse optimizer endpoints are deliberately retained as other branches.
    """
    snapshot = _frame(snapshot)
    _coords(snapshot)
    bank = {str(key): [] for key in sorted(snapshot.index)}
    seen = {key: set() for key in bank}

    def add(key, theta, chi2, origin):
        key = str(key)
        if key not in bank:
            raise ValueError(f"Seed target absent from snapshot: {key}")
        entry = _entry(snapshot.loc[key], key, theta, chi2, origin)
        token = tuple(float(v).hex() for v in entry["theta"])
        if token not in seen[key]:
            bank[key].append(entry)
            seen[key].add(token)

    for key in sorted(bank):
        row = snapshot.loc[key]
        add(key, [row["parameter_" + n] for n in NAMES], row["chi2_one_loop"], "baseline")
    best = _table(best_csv)
    if not best.empty:
        if "target" not in best or best.target.duplicated().any():
            raise ValueError("Best seed targets must be present and unique")
        for row in best.sort_values("target", kind="stable").to_dict("records"):
            key = str(row["target"])
            if key not in bank:
                raise ValueError(f"Unknown best-seed key: {key}")
            _identity(row, snapshot.loc[key], key)
            add(key, [row["parameter_best_" + n] for n in NAMES], row["chi2_best"], "seed_best")
    attempts = _table(attempts_csv)
    if not attempts.empty:
        if "trial_id" not in attempts or attempts.trial_id.duplicated().any():
            raise ValueError("Attempt trial_id must be present and unique")
        for row in attempts.sort_values(["target", "trial_id"], kind="stable").to_dict("records"):
            key = str(row["target"])
            if key not in bank:
                raise ValueError(f"Unknown attempt target: {key}")
            _identity(row, snapshot.loc[key], key)
            for suffix in ("selected", "optimizer_final"):
                field = "theta_" + suffix
                value = row.get(field)
                if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                    continue
                if isinstance(value, str) and value.strip().lower() in ("", "null", "nan", "none"):
                    continue
                add(key, value, row["chi2_" + suffix], f"seed_attempt:{row['trial_id']}:{suffix}")
    return bank


def _q_token(entry):
    theta = _theta(entry["theta"])
    # Exact four-dimensional identity; no absolute tolerance on tiny alpha_ct.
    return tuple(float(theta[i]).hex() for i in Q_INDICES)


def _stable_entries(bank):
    entries = []
    for key in sorted(bank):
        for entry in bank[key]:
            copied = deepcopy(entry)
            copied.setdefault("source_key", str(key))
            entries.append(copied)
    entries.sort(key=lambda e: (str(e["source_key"]), _q_token(e), str(e.get("origin", ""))))
    unique, seen = [], set()
    for entry in entries:
        token = _q_token(entry)
        if token not in seen:
            seen.add(token)
            unique.append(entry)
    return unique


def global_representatives(bank, limit=8):
    """Diverse shape representatives, never ranked by cross-target raw chi2.

    Cover available sign combinations of alpha_bias and beta_ct, then use
    deterministic farthest-point sampling in scaled four-parameter space.
    At most two representatives come from any one source coordinate.
    """
    if int(limit) != limit or limit < 0:
        raise ValueError("Representative limit must be a nonnegative integer")
    entries = _stable_entries(bank)
    if not entries or limit == 0:
        return []
    vectors = np.array([np.asarray(e["theta"])[list(Q_INDICES)] / Q_SCALES for e in entries])
    signs = [(int(np.sign(v[1])), int(np.sign(v[3]))) for v in vectors]
    selected, counts = [], defaultdict(int)

    def eligible(i):
        return i not in selected and counts[str(entries[i]["source_key"])] < 2

    def take(i):
        selected.append(i)
        counts[str(entries[i]["source_key"]) ] += 1

    # Nonzero sign combinations first; exact-zero faces follow when present.
    groups = sorted(set(signs), key=lambda s: (0 in s, s))
    for group in groups:
        if len(selected) >= limit:
            break
        ids = [i for i, sign in enumerate(signs) if sign == group and eligible(i)]
        if not ids:
            continue
        median = np.median(vectors[ids], axis=0)
        take(min(ids, key=lambda i: (float(np.sum((vectors[i] - median) ** 2)), i)))
    while len(selected) < limit:
        ids = [i for i in range(len(entries)) if eligible(i)]
        if not ids:
            break
        def distance(i):
            return min(float(np.sum((vectors[i] - vectors[j]) ** 2)) for j in selected)
        take(max(ids, key=lambda i: (distance(i), -i)))
    return [deepcopy(entries[i]) for i in selected]


def proposals_for(key, bank, neighbor_map, global_bank):
    """Own, all neighbouring and global proposals, deduplicated by exact q.

    No target scoring or raw-chi2 truncation occurs here. The target worker
    must first enforce its own q domain and profile its own amplitudes.
    """
    key = str(key)
    if key not in bank:
        raise KeyError(f"Unknown target key: {key}")
    entries = []
    for source in [key, *neighbor_map.get(key, [])]:
        if source not in bank:
            raise KeyError(f"Neighbour {source} has no candidate bank")
        for member in bank[source]:
            copied = deepcopy(member)
            copied.setdefault("source_key", str(source))
            entries.append(copied)
    entries.extend(global_bank)
    result, seen = [], set()
    for entry in entries:
        token = _q_token(entry)
        if token not in seen:
            seen.add(token)
            result.append(deepcopy(entry))
    return result


def _improved_keys(best_keys):
    if isinstance(best_keys, pd.DataFrame):
        frame = best_keys.copy(deep=True)
        if "gain" not in frame:
            frame["gain"] = frame.chi2_old - frame.chi2_best
        frame = frame[frame.gain > .01].sort_values(["gain", "target"], ascending=[False, True])
        return frame.target.astype(str).tolist()
    if isinstance(best_keys, dict):
        return [str(k) for k, v in sorted(best_keys.items(), key=lambda item: (-float(item[1]), str(item[0])))
                if float(v) > .01]
    return [str(key) for key in best_keys]


def pilot_keys(snapshot, upper_keys, control_keys, best_keys, n=32):
    """Deterministic pilot with controls, old minimum, anchors and improvements.

    best_keys may be a best-results DataFrame, key->gain mapping, or an ordered
    list of improved keys. A list is assumed already ordered by decreasing gain.
    Coverage uses recorded momentum-bin/source-label groups, without inferring
    unknown CLASS settings from directory names; exact metadata is a worker gate.
    """
    snapshot = _frame(snapshot)
    coordinates = _coords(snapshot)
    upper = {str(k) for k in upper_keys}
    controls = list(dict.fromkeys(str(k) for k in control_keys))
    allowed = upper | set(controls)
    if not allowed.issubset(coordinates):
        raise ValueError("Pilot keys contain coordinates outside the snapshot")
    if n < 1 or int(n) != n or len(allowed) < n:
        raise ValueError("Pilot size must be positive and no larger than the requested domain")
    selected = []

    def add(key, *, mandatory=False):
        key = str(key)
        if key not in allowed:
            if mandatory:
                raise ValueError(f"Mandatory pilot point outside requested domain: {key}")
            return
        if key not in selected:
            selected.append(key)

    for key in controls:
        add(key, mandatory=True)
    global_min = min(coordinates, key=lambda k: (float(snapshot.loc[k, "chi2_one_loop"]), k))
    add(global_min, mandatory=True)
    for key in ANCHORS:
        add(key, mandatory=True)
    for key in _improved_keys(best_keys)[:9]:
        add(key, mandatory=True)
    if len(selected) > n:
        raise ValueError("Pilot size cannot contain all mandatory controls/anchors/improvements")

    # Include all source/numerical categories represented in the requested set.
    group_columns = [c for c in ("momentum_bins", "source_label", "class_params_json",
                                 "actual_class_params_json") if c in snapshot]
    groups = defaultdict(list)
    for key in sorted(allowed):
        row = snapshot.loc[key]
        group = tuple(str(row[c]) for c in group_columns)
        groups[group].append(key)
    for group in sorted(groups):
        ids = groups[group]
        if any(key in selected for key in ids) or len(selected) >= n:
            continue
        add(ids[len(ids) // 2])

    values = np.array([coordinates[k] for k in sorted(allowed)])
    span = np.maximum(np.ptp(values, axis=0), 1.)
    points = {k: np.array(coordinates[k]) / span for k in allowed}
    while len(selected) < n:
        remaining = sorted(allowed - set(selected))
        def distance(key):
            return min(float(np.sum((points[key] - points[other]) ** 2)) for other in selected)
        add(max(remaining, key=lambda key: (distance(key), key)))
    return selected
