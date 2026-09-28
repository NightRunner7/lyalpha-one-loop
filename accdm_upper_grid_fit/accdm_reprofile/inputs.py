"""Read-only, provenance-checked inputs for the accDM profiling pilot.

The frozen CSV is an index of existing results, not a replacement for the
project's JSON/NPZ/data files. A context is usable only after the canonical
likelihood reproduces the original objective without rescaling it.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


PARAMETER_NAMES = (
    "log_alpha_F", "beta_F", "alpha_bias", "beta_bias", "alpha_ct", "beta_ct"
)
LEGACY_ROOTS = (
    Path("/home/2/ks405818/Master/lyalpha_one_loop"),
    Path("/home/krzysztof/Workspace/lyalpha_one_loop"),
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE_ROOT / "pilot_points.csv"
BUNDLED_SNAPSHOT = PACKAGE_ROOT / "baseline_selected_raw_points.csv"
DIAGONAL_ATOL = 1.e-4
DIAGONAL_RTOL = 1.e-7


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_key(logm: float, logf: float) -> str:
    """The native campaign identity uses exactly ten decimal places."""
    values = [float(logm), float(logf)]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite native coordinates.")
    return "|".join("0.0000000000" if abs(x) < 0.5e-10 else f"{x:.10f}"
                    for x in values)


def resolve_path(raw: str | Path, project_root: str | Path) -> Path:
    """Remap only explicitly known roots; never search by a bare filename.

    Recognized legacy paths always map to the requested project, even if an
    old file still exists. Other absolute paths are preserved and reported.
    """
    if raw is None or (not isinstance(raw, (str, Path)) and pd.isna(raw)):
        raise ValueError("Missing saved path.")
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        raise ValueError("Missing saved path.")
    root = Path(project_root).expanduser().resolve()
    path = Path(text).expanduser()
    if not path.is_absolute():
        return (root / path).resolve()
    for previous in LEGACY_ROOTS:
        try:
            relative = path.relative_to(previous)
        except ValueError:
            continue
        return (root / relative).resolve()
    return path.resolve()


def _read_coordinate_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path, dtype={"coordinate_key": str})
    required = {"coordinate_key", "log10m_acc", "log10f_acc"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: empty input table.")
    expected = [coordinate_key(m, f) for m, f in
                zip(frame.log10m_acc, frame.log10f_acc)]
    if frame.coordinate_key.tolist() != expected:
        raise ValueError(f"{path}: coordinate_key differs from native coordinates at 10 decimals.")
    if frame.coordinate_key.duplicated().any():
        duplicates = frame.loc[frame.coordinate_key.duplicated(), "coordinate_key"].tolist()
        raise ValueError(f"{path}: duplicate coordinate identities: {duplicates[:5]}")
    frame = frame.set_index("coordinate_key", drop=False)
    frame.index.name = "coordinate_index"
    frame.attrs.update(path=str(path), sha256=sha256_file(path))
    return frame


def load_snapshot(path: str | Path) -> pd.DataFrame:
    frame = _read_coordinate_table(path)
    required = {"fit_path", "theory_path", "chi2_one_loop", "theory_digest",
                "k_uv_cut_hmpc", "covariance_mode"}
    required.update(prefix + name for prefix in ("parameter_", "lower_", "upper_")
                    for name in PARAMETER_NAMES)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Snapshot lacks required fitted quantities: {sorted(missing)}. "
                         "Use the full selected_raw_points.csv export.")
    frame.attrs.update(snapshot_path=frame.attrs["path"],
                       snapshot_sha256=frame.attrs["sha256"])
    return frame


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    frame = _read_coordinate_table(path)
    frame.attrs.update(manifest_path=frame.attrs["path"],
                       manifest_sha256=frame.attrs["sha256"])
    return frame


def select_pilot(snapshot: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    missing = manifest.index.difference(snapshot.index)
    if len(missing):
        raise ValueError(f"Snapshot is missing {len(missing)} pilot points: {missing.tolist()}")
    selected = snapshot.loc[manifest.index].copy()
    for column in manifest:
        if column not in {"coordinate_key", "log10m_acc", "log10f_acc"}:
            selected["pilot_" + column] = manifest[column]
    selected.attrs.update(snapshot.attrs)
    selected.attrs.update(manifest.attrs)
    return selected


def discover_snapshot(project_root: str | Path) -> Path:
    """Newest full diagnostic export, otherwise the bundled frozen snapshot.

    A malformed newer export is not silently replaced by an older snapshot:
    load_snapshot/load_context will explain the failed check to the caller.
    """
    parent = Path(project_root).expanduser().resolve() / "runs/accdm/analysis_neighbor_diagnostics"
    candidates = ([parent / "selected_raw_points.csv"] if
                  (parent / "selected_raw_points.csv").is_file() else [])
    if parent.is_dir():
        candidates += list(parent.glob("*/selected_raw_points.csv"))
    if candidates:
        return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))).resolve()
    if BUNDLED_SNAPSHOT.is_file():
        return BUNDLED_SNAPSHOT
    raise FileNotFoundError("No selected_raw_points.csv found; pass --snapshot explicitly.")


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, (float, np.floating)) and np.isnan(value))


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid {label}.") from error
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite {label}.")
    return result


def _saved_bounds(record: Mapping[str, Any]) -> np.ndarray:
    rows = record.get("boundary_diagnostics", [])
    names = [item.get("parameter") for item in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate nuisance entries in boundary_diagnostics.")
    by_name = {item["parameter"]: item for item in rows}
    missing = set(PARAMETER_NAMES) - set(by_name)
    if missing:
        raise ValueError(f"Missing saved nuisance bounds: {sorted(missing)}")
    bounds = np.array([[by_name[name]["lower"], by_name[name]["upper"]]
                       for name in PARAMETER_NAMES], dtype=float)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid saved nuisance bounds.")
    return bounds


def nuisance_is_admissible(context: Mapping[str, Any], theta: Any,
                           tolerance: float = 1.e-10) -> bool:
    theta = np.asarray(theta, dtype=float)
    bounds = np.asarray(context["bounds"], dtype=float)
    return bool(theta.shape == (6,) and np.isfinite(theta).all()
                and np.all(theta >= bounds[:, 0] - tolerance)
                and np.all(theta <= bounds[:, 1] + tolerance))


def _project_api(root: Path) -> tuple[Any, Any, Any, dict[str, str]]:
    if not (root / "lyalpha_pt").is_dir():
        raise FileNotFoundError(f"Missing canonical project code: {root / 'lyalpha_pt'}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    data_module = importlib.import_module("lyalpha_pt.data")
    fit_module = importlib.import_module("lyalpha_pt.fit")
    theory_module = importlib.import_module("lyalpha_pt.theory")
    if tuple(fit_module.PARAMETER_NAMES) != PARAMETER_NAMES:
        raise ValueError(f"Canonical API nuisance order differs: {fit_module.PARAMETER_NAMES}")
    functions = {"fit": fit_module.EffectiveModelFit,
                 "data": data_module.load_dr12, "theory": theory_module.load_theory}
    sources = {}
    for name, function in functions.items():
        source = Path(inspect.getfile(function)).resolve()
        if root not in source.parents:
            raise RuntimeError(f"{name} imported outside requested project: {source}. "
                               "Restart Python with the intended --project-root.")
        sources[name + "_path"] = str(source)
        sources[name + "_sha256"] = sha256_file(source)
    return (fit_module.EffectiveModelFit, data_module.load_dr12,
            theory_module.load_theory, sources)


def _array_hash(array: Any) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_context(project_root: str | Path, row: Mapping[str, Any],
                 data_dir_override: str | Path | None = None) -> dict[str, Any]:
    """Validate a baseline and construct the unchanged canonical likelihood.

    No theory generation, smoothing, fitting, parameter clipping or covariance
    rescaling occurs here. Stale CSV/JSON combinations fail before profiling.
    """
    root = Path(project_root).expanduser().resolve()
    row = dict(row)
    key = coordinate_key(row["log10m_acc"], row["log10f_acc"])
    if row.get("coordinate_key") != key:
        raise ValueError("Row coordinate identity is inconsistent.")
    fit_path = resolve_path(row.get("fit_path"), root)
    theory_path = resolve_path(row.get("theory_path"), root)
    for label, path in (("Fit JSON", fit_path), ("Theory NPZ", theory_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    saved = json.loads(fit_path.read_text(encoding="utf-8"))
    if "one_loop" not in saved.get("fits", {}):
        raise ValueError(f"{fit_path}: missing fits.one_loop.")
    record = saved["fits"]["one_loop"]
    if record.get("mode", "one_loop") != "one_loop":
        raise ValueError("Saved fit mode is not one_loop.")
    theta = np.array([record["parameters"][name] for name in PARAMETER_NAMES], dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError("Nonfinite saved nuisance parameters.")
    bounds = _saved_bounds(record)
    chi2_saved = _finite_number(record.get("chi2"), "saved chi2")
    snapshot_chi2 = _finite_number(row.get("chi2_one_loop"), "snapshot chi2")
    if not np.isclose(chi2_saved, snapshot_chi2, atol=DIAGONAL_ATOL, rtol=DIAGONAL_RTOL):
        raise ValueError("Current fit JSON chi2 differs from snapshot; supply a consistent snapshot.")
    for i, name in enumerate(PARAMETER_NAMES):
        old = _finite_number(row.get("parameter_" + name), "snapshot " + name)
        # Relative-only comparison preserves tiny nonzero CT amplitudes such
        # as 1e-87: an absolute 1e-8 tolerance would silently treat them as zero.
        if not np.isclose(old, theta[i], atol=0., rtol=1.e-9):
            raise ValueError(f"Current JSON nuisance {name} differs from snapshot.")
        previous = np.array([_finite_number(row.get(side + "_" + name),
                                            f"snapshot {side} {name}")
                             for side in ("lower", "upper")])
        if not np.allclose(bounds[i], previous, rtol=1.e-12, atol=1.e-12):
            raise ValueError(f"Current JSON bounds for {name} differ from snapshot.")
    if not nuisance_is_admissible({"bounds": bounds}, theta):
        raise ValueError("Saved nuisance parameters lie outside their saved bounds.")
    cutoff = _finite_number(record.get("k_uv_cut_hmpc"), "saved k_uv_cut_hmpc")
    if cutoff <= 0:
        raise ValueError("Saved UV cutoff must be positive.")
    if not np.isclose(cutoff, _finite_number(row.get("k_uv_cut_hmpc"), "snapshot cutoff"),
                      rtol=1.e-12, atol=1.e-12):
        raise ValueError("Saved UV cutoff differs from snapshot.")
    covariance_mode = record.get("covariance_mode")
    if not isinstance(covariance_mode, str) or not covariance_mode:
        raise ValueError("Missing covariance_mode in saved fit JSON.")
    if covariance_mode != row.get("covariance_mode"):
        raise ValueError("Saved covariance_mode differs from snapshot.")
    fit_digest = record.get("theory_digest")
    if not fit_digest or fit_digest != row.get("theory_digest"):
        raise ValueError("Current fit theory_digest differs from snapshot or is missing.")
    if _present(saved.get("theory_file")):
        declared = resolve_path(saved["theory_file"], root)
        if declared != theory_path:
            raise ValueError(f"Fit JSON theory_file differs from snapshot: {declared} != {theory_path}")
    data_raw = data_dir_override if data_dir_override is not None else saved.get("data_dir")
    data_dir = resolve_path(data_raw, root)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DR12 data directory not found: {data_dir}; use --data-dir if relocated.")
    EffectiveModelFit, load_dr12, load_theory, sources = _project_api(root)
    theory = load_theory(theory_path)
    meta = theory.metadata
    digest = meta.get("bundle_digest")
    if not digest or digest != fit_digest:
        raise ValueError("Fit JSON theory_digest and NPZ bundle_digest do not match.")
    dataset = load_dr12(data_dir)
    if "n_data" not in record:
        raise ValueError("Missing n_data in saved fit JSON.")
    if dataset.n_data != int(record["n_data"]):
        raise ValueError("DR12 data count differs from saved fit JSON.")
    z_data = np.asarray(dataset.z_unique, dtype=float)
    z_theory = np.asarray(theory.z, dtype=float)
    if (z_data.shape != z_theory.shape or not np.isfinite(z_theory).all()
            or not np.allclose(z_data, z_theory, rtol=1.e-10, atol=1.e-10)):
        raise ValueError("Data and theory redshifts differ.")
    per_z = record.get("chi2_by_redshift")
    if per_z:
        saved_z = np.array([item["z"] for item in per_z], dtype=float)
        if saved_z.shape != z_data.shape or not np.allclose(saved_z, z_data, atol=1.e-10, rtol=1.e-10):
            raise ValueError("Saved fit redshifts differ from loaded data.")
    k_loop = np.asarray(theory.k_loop, dtype=float)
    if (k_loop.ndim != 1 or len(k_loop) < 2 or not np.isfinite(k_loop).all()
            or np.any(k_loop <= 0) or np.any(np.diff(k_loop) <= 0)):
        raise ValueError("Invalid one-loop k grid.")
    if not (k_loop[0] < cutoff <= k_loop[-1] * (1. + 1.e-12)):
        raise ValueError("Saved UV cutoff lies outside the one-loop k coverage.")
    fitter = EffectiveModelFit(dataset, theory, mode="one_loop", k_uv_cut=cutoff,
                               covariance_mode=covariance_mode)
    covariance = np.asarray(fitter.covariance, dtype=float)
    if (covariance.shape != (dataset.n_data, dataset.n_data)
            or not np.isfinite(covariance).all()
            or np.any(np.diag(covariance) <= 0)
            or not np.allclose(covariance, covariance.T, atol=1.e-12, rtol=1.e-10)):
        raise ValueError("Canonical likelihood covariance is invalid.")
    expected_covariance = np.asarray(dataset.covariance(covariance_mode), dtype=float)
    if not np.array_equal(covariance, expected_covariance):
        raise ValueError("Fitter covariance differs from canonical dataset.covariance(mode); "
                         "unrecorded covariance rescaling is not allowed.")
    reconstructed = float(fitter.chi2(theta))
    if not np.isfinite(reconstructed) or not np.isclose(
            reconstructed, chi2_saved, atol=DIAGONAL_ATOL, rtol=DIAGONAL_RTOL):
        raise RuntimeError(f"Canonical chi2 roundtrip failed for {key}: saved={chi2_saved:.12g}, "
                           f"recomputed={reconstructed:.12g}. Profiling blocked; check code/data/settings.")
    prediction = np.asarray(fitter.model(theta), dtype=float)
    if prediction.shape != (dataset.n_data,) or not np.isfinite(prediction).all():
        raise ValueError("Saved nuisance yield an invalid P1D prediction.")
    # Preserve actual metadata, never infer CLASS precision from campaign tags.
    model_metadata = meta.get("model", {})
    # The canonical generator writes the actual CLASS input at top level.
    # model.class_params is the model specification before runtime additions.
    actual_class_params = meta.get("class_params", {})
    model_class_params = model_metadata.get("class_params", {}) if isinstance(model_metadata, dict) else {}
    metadata_numerics = meta.get("numerics", {})
    overrides = {} if data_dir_override is None else {"data_dir": str(data_dir)}
    validation = {
        "coordinate_key": key, "status": "validated", "chi2_saved": chi2_saved,
        "chi2_recomputed": reconstructed, "difference": reconstructed - chi2_saved,
        "diagonal_atol": DIAGONAL_ATOL, "diagonal_rtol": DIAGONAL_RTOL,
        "fit_path": str(fit_path), "fit_sha256": sha256_file(fit_path),
        "theory_path": str(theory_path), "theory_sha256": sha256_file(theory_path),
        "theory_digest": digest, "data_dir": str(data_dir), "n_data": int(dataset.n_data),
        "redshifts": z_data.tolist(), "cutoff": cutoff, "covariance": covariance_mode,
        "covariance_sha256": _array_hash(covariance), "overrides": overrides,
        "source_path": sources["fit_path"], "source_sha256": sources["fit_sha256"],
        "canonical_sources": sources, "actual_class_params": actual_class_params,
        "class_params_metadata_location": "class_params" if "class_params" in meta else "missing",
        "model_class_params": model_class_params,
        "theory_numerics": metadata_numerics,
    }
    for name in ("z", "k_velocity", "p1d"):
        if hasattr(dataset, name):
            validation["data_" + name + "_sha256"] = _array_hash(getattr(dataset, name))
    return {
        "fitter": fitter, "theory": theory, "dataset": dataset, "theta": theta,
        "bounds": bounds, "chi2_saved": chi2_saved, "chi2_recomputed": reconstructed,
        "row": row, "record": record, "saved": saved, "validated": True,
        "fit_path": str(fit_path), "theory_path": str(theory_path), "data_dir": str(data_dir),
        "mode": "one_loop", "config": {"k_uv_cut_hmpc": cutoff, "covariance_mode": covariance_mode},
        "overrides": overrides, "source_path": sources["fit_path"],
        "source_sha256": sources["fit_sha256"], "validation_metadata": validation,
        "metadata_numerics": metadata_numerics, "actual_class_params": actual_class_params,
    }
