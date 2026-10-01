#!/usr/bin/env python3
"""Build an immutable DCDM model manifest for a resumable grid campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from lyalpha_pt.models import ModelSpec
from cluster.config import scheduler_config


MPC_IN_KM = 3.0856775814913673e19
GYR_IN_SECONDS = 1.0e9 * 365.25 * 24.0 * 3600.0
GAMMA_TAU_CONVERSION = MPC_IN_KM / GYR_IN_SECONDS

MANIFEST_FIELDS = (
    "index",
    "point_id",
    "active",
    "grid_level",
    "tau_gyr",
    "Gamma_dcdm",
    "epsilon_dcdm",
    "model_json",
    "model_hash",
)


def gamma_from_tau_gyr(tau_gyr: float) -> float:
    """Convert a DCDM lifetime in Gyr to Gamma_dcdm in km/s/Mpc."""

    tau = float(tau_gyr)
    if not tau > 0.0:
        raise ValueError("tau_gyr must be positive.")
    return GAMMA_TAU_CONVERSION / tau


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _number_token(value: float) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def point_id(tau_gyr: float, epsilon_dcdm: float) -> str:
    return f"dcdm_tau{_number_token(tau_gyr)}_eps{_number_token(epsilon_dcdm)}"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def _expand_points(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    has_points = "points" in spec
    has_axes = "axes" in spec
    if has_points == has_axes:
        raise ValueError("Grid spec must contain exactly one of 'points' or 'axes'.")

    if has_points:
        raw_points = spec["points"]
        if not isinstance(raw_points, list):
            raise ValueError("'points' must be a list.")
        points = [dict(item) for item in raw_points]
    else:
        axes = spec["axes"]
        if not isinstance(axes, dict):
            raise ValueError("'axes' must be an object.")
        tau_values = axes.get("tau_gyr")
        epsilon_values = axes.get("epsilon_dcdm")
        if not isinstance(tau_values, list) or not isinstance(epsilon_values, list):
            raise ValueError("Both DCDM axes must be lists.")
        points = [
            {
                "tau_gyr": tau,
                "epsilon_dcdm": epsilon,
                "grid_level": spec.get("grid_level", 0),
                "active": True,
            }
            for tau, epsilon in itertools.product(tau_values, epsilon_values)
        ]

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in points:
        if "tau_gyr" not in raw or "epsilon_dcdm" not in raw:
            raise ValueError("Every point needs tau_gyr and epsilon_dcdm.")
        tau = float(raw["tau_gyr"])
        epsilon = float(raw["epsilon_dcdm"])
        if not tau > 0.0:
            raise ValueError(f"Invalid tau_gyr={tau}.")
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"Invalid epsilon_dcdm={epsilon}; expected 0 < epsilon < 1.")
        identifier = point_id(tau, epsilon)
        if identifier in seen:
            raise ValueError(f"Duplicate DCDM coordinate: {identifier}.")
        seen.add(identifier)
        normalized.append(
            {
                "point_id": identifier,
                "tau_gyr": tau,
                "epsilon_dcdm": epsilon,
                "grid_level": int(raw.get("grid_level", spec.get("grid_level", 0))),
                "active": bool(raw.get("active", True)),
            }
        )
    return normalized


def _model_for_point(base: Mapping[str, Any], point: Mapping[str, Any]) -> ModelSpec:
    params = dict(base["class_params"])
    params["Gamma_dcdm"] = gamma_from_tau_gyr(float(point["tau_gyr"]))
    params["epsilon_dcdm"] = float(point["epsilon_dcdm"])

    forbidden = {"output", "P_k_max_h/Mpc", "z_max_pk"}.intersection(params)
    if forbidden:
        raise ValueError(
            "The base model must not override generator-owned CLASS keys: "
            + ", ".join(sorted(forbidden))
        )
    if params.get("omega_cdm") == 0.1200:
        raise ValueError(
            "DCDM base model still contains the LCDM omega_cdm=0.1200 density. "
            "Use omega_ini_dcdm2 for the decaying component."
        )
    if "omega_ini_dcdm2" not in params:
        raise ValueError("DCDM base model is missing omega_ini_dcdm2.")

    tags = list(base.get("tags", []))
    tags.extend(
        [
            f"tau-gyr:{point['tau_gyr']:g}",
            f"epsilon-dcdm:{point['epsilon_dcdm']:g}",
        ]
    )
    description = (
        f"{base.get('description', 'DCDM model')} "
        f"tau={point['tau_gyr']:g} Gyr, epsilon={point['epsilon_dcdm']:g}."
    )
    model = ModelSpec(
        name=str(point["point_id"]),
        description=description,
        class_params=params,
        loop_source=str(base.get("loop_source", "total")),
        loop_weight=base.get("loop_weight", 1.0),
        tags=tuple(tags),
    )
    model.validate()
    return model


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(rows[0]) != set(MANIFEST_FIELDS):
        raise ValueError(f"Existing manifest has unexpected columns: {sorted(rows[0])}")
    indices = [int(row["index"]) for row in rows]
    point_ids = [row["point_id"] for row in rows]
    if len(indices) != len(set(indices)) or len(point_ids) != len(set(point_ids)):
        raise ValueError("Existing manifest has duplicate indices or point IDs.")
    return sorted(rows, key=lambda row: int(row["index"]))


def _write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_campaign(base_model_path: Path, grid_spec_path: Path, run_dir: Path) -> dict[str, Any]:
    """Create or safely extend one run directory and return its campaign config."""

    base_model_path = base_model_path.resolve()
    grid_spec_path = grid_spec_path.resolve()
    run_dir = run_dir.resolve()
    base = _load_json_object(base_model_path)
    spec = _load_json_object(grid_spec_path)
    points = _expand_points(spec)

    required_base = {"class_params", "loop_source", "loop_weight"}
    missing_base = required_base.difference(base)
    if missing_base:
        raise ValueError(f"Base model is missing keys: {sorted(missing_base)}")
    if not str(spec.get("campaign_id", "")).strip():
        raise ValueError("Grid spec needs a non-empty campaign_id.")

    existing_config_path = run_dir / "campaign.json"
    if existing_config_path.exists():
        existing_config = _load_json_object(existing_config_path)
        if existing_config.get("campaign_id") != spec["campaign_id"]:
            raise ValueError(
                "Refusing to change campaign_id inside an existing run directory."
            )
        if existing_config.get("base_model_hash") != _digest(base):
            raise ValueError(
                "Refusing to change the base model inside an existing run directory. "
                "Create a new campaign instead."
            )

    for relative in (
        "models",
        "theory",
        "fits",
        "checkpoints",
        "status/theory",
        "status/fit",
        "jobs/theory",
        "jobs/fit",
        "logs/theory",
        "logs/fit",
        "controller",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.csv"
    rows = _read_manifest(manifest_path)
    by_point = {row["point_id"]: row for row in rows}
    next_index = max((int(row["index"]) for row in rows), default=-1) + 1

    for point in points:
        model = _model_for_point(base, point)
        payload = model.as_dict()
        model_hash = _digest(payload)
        relative_model_path = f"models/{point['point_id']}.json"
        model_path = run_dir / relative_model_path
        if model_path.exists():
            existing_payload = _load_json_object(model_path)
            if _digest(existing_payload) != model_hash:
                raise ValueError(
                    f"Refusing to change immutable model file {model_path}. "
                    "Use a new campaign directory after changing the base model."
                )
        else:
            _atomic_json(model_path, payload)

        existing_row = by_point.get(str(point["point_id"]))
        if existing_row is not None:
            if existing_row["model_hash"] != model_hash:
                raise ValueError(
                    f"Manifest/model hash mismatch for {point['point_id']}; "
                    "use a new campaign directory."
                )
            continue

        row = {
            "index": next_index,
            "point_id": point["point_id"],
            "active": int(point["active"]),
            "grid_level": point["grid_level"],
            "tau_gyr": f"{point['tau_gyr']:.16g}",
            "Gamma_dcdm": f"{gamma_from_tau_gyr(point['tau_gyr']):.16g}",
            "epsilon_dcdm": f"{point['epsilon_dcdm']:.16g}",
            "model_json": relative_model_path,
            "model_hash": model_hash,
        }
        rows.append(row)
        by_point[str(point["point_id"])] = row
        next_index += 1

    rows.sort(key=lambda row: int(row["index"]))
    _write_manifest(manifest_path, rows)

    theory = dict(spec.get("theory", {}))
    fit = dict(spec.get("fit", {}))
    cluster = dict(spec.get("cluster", {}))
    quality = theory.get("quality", "production")
    if quality not in {"smoke", "production", "precision"}:
        raise ValueError(f"Unsupported theory quality: {quality!r}.")
    mode = fit.get("mode", "one_loop")
    if mode not in {"linear", "one_loop", "both"}:
        raise ValueError(f"Unsupported fit mode: {mode!r}.")
    fit_strategy = str(fit.get("strategy", "standard"))
    if fit_strategy not in {"standard", "neighbor_refit"}:
        raise ValueError(f"Unsupported fit strategy: {fit_strategy!r}.")
    source_fit_dirs = [str(value) for value in fit.get("source_fit_dirs", [])]
    if fit_strategy == "neighbor_refit" and not source_fit_dirs:
        raise ValueError(
            "fit.strategy='neighbor_refit' requires at least one source_fit_dirs entry."
        )

    config = {
        "schema_version": 1,
        "campaign_id": spec["campaign_id"],
        "description": spec.get("description", ""),
        "model_family": "dcdm",
        "grid_axes": ["tau_gyr", "epsilon_dcdm"],
        "manifest": "manifest.csv",
        "data_dir": "data",
        "base_model_name": base_model_path.name,
        "base_model_hash": _digest(base),
        "grid_spec_name": grid_spec_path.name,
        "grid_spec_hash": _digest(spec),
        "point_count": len(rows),
        "active_point_count": sum(_truthy(row["active"]) for row in rows),
        "theory": {
            "quality": quality,
            "max_attempts": int(theory.get("max_attempts", 3)),
        },
        "fit": {
            "mode": mode,
            "strategy": fit_strategy,
            "source_fit_dirs": source_fit_dirs,
            "k_uv_cut": float(fit.get("k_uv_cut", 20.0)),
            "covariance": fit.get("covariance", "paper_diag"),
            "seeds": [int(value) for value in fit.get("seeds", [12345, 23456, 34567])],
            "expanded_counterterm": bool(fit.get("expanded_counterterm", True)),
            "full_six_dimensional": bool(fit.get("full_six_dimensional", False)),
            "no_cutoff_continuation": bool(fit.get("no_cutoff_continuation", False)),
            "de_maxiter": int(fit.get("de_maxiter", 300)),
            "de_popsize": int(fit.get("de_popsize", 20)),
            "max_attempts": int(fit.get("max_attempts", 3)),
        },
        "cluster": {
            **scheduler_config(cluster),
            "conda_exe": cluster.get("conda_exe", "/opt/anaconda3/bin/conda"),
            "theory_env": cluster.get("theory_env", "class_decays"),
            "fit_env": cluster.get("fit_env", cluster.get("theory_env", "class_decays")),
            "theory_python": cluster.get("theory_python", ""),
            "fit_python": cluster.get("fit_python", ""),
            "theory_ncpus": int(cluster.get("theory_ncpus", 1)),
            "theory_threads": int(
                cluster.get("theory_threads", cluster.get("theory_ncpus", 1))
            ),
            "theory_mem": str(cluster.get("theory_mem", "8gb")),
            "fit_ncpus": int(cluster.get("fit_ncpus", 1)),
            "fit_mem": str(cluster.get("fit_mem", "4gb")),
            "theory_max_active": int(cluster.get("theory_max_active", 50)),
            "fit_max_active": int(cluster.get("fit_max_active", 100)),
            "max_user_active": int(cluster.get("max_user_active", 300)),
            "poll_seconds": float(cluster.get("poll_seconds", 30.0)),
            "submit_delay_seconds": float(cluster.get("submit_delay_seconds", 0.2)),
        },
    }
    _atomic_json(run_dir / "campaign.json", config)
    return config


def _default_base_model() -> Path:
    return Path(__file__).with_name("base_model.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=_default_base_model())
    parser.add_argument("--grid-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    config = build_campaign(args.base_model, args.grid_spec, args.run_dir)
    run_dir = args.run_dir.resolve()
    print(f"campaign: {config['campaign_id']}")
    print(f"points: {config['point_count']} ({config['active_point_count']} active)")
    print(f"manifest: {run_dir / 'manifest.csv'}")
    print(f"configuration: {run_dir / 'campaign.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
