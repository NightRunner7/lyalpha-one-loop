#!/usr/bin/env python3
"""Freeze and build q=5001 recovery campaigns from an incomplete high-f run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from campaigns.accdm.build_grid import build_campaign


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCDM_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "runs" / "accdm" / "accdm_scan_highf_q5001_v1"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "accdm"


@dataclass(frozen=True)
class RecoveryRun:
    role: str
    campaign_id: str
    base_model: Path
    points: tuple[dict[str, Any], ...]
    theory_ncpus: int
    theory_threads: int
    theory_mem: str
    theory_max_active: int


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_rows(source: Path) -> list[dict[str, str]]:
    manifest = source / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"index", "point_id", "active", "log10m_acc", "log10f_acc"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid or empty source manifest: {manifest}")
    return sorted(rows, key=lambda row: int(row["index"]))


def _is_active(row: Mapping[str, str]) -> bool:
    return str(row["active"]).strip().lower() in {"1", "true", "yes", "y", "on"}


def _theory_complete(source: Path, point_id: str) -> bool:
    return (
        (source / "theory" / f"{point_id}.npz").is_file()
        and (source / "status" / "theory" / f"{point_id}.done.json").is_file()
    )


def _failure_text(source: Path, point_id: str) -> str:
    paths = (
        source / "logs" / "theory" / f"{point_id}.out",
        source / "logs" / "theory" / f"{point_id}.err",
        source / "status" / "theory" / f"{point_id}.failed.json",
    )
    chunks: list[str] = []
    for path in paths:
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def classify_source_points(
    source: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], int]:
    """Return ordinary missing points, stiff points, and completed count."""

    ordinary: list[dict[str, Any]] = []
    stiff: list[dict[str, Any]] = []
    completed = 0
    for row in _source_rows(source):
        if not _is_active(row):
            continue
        point_id = row["point_id"]
        if _theory_complete(source, point_id):
            completed += 1
            continue
        point = {
            "log10m_acc": float(row["log10m_acc"]),
            "log10f_acc": float(row["log10f_acc"]),
            "grid_level": int(row.get("grid_level", 0)),
            "active": True,
        }
        if "step size too small" in _failure_text(source, point_id).lower():
            stiff.append(point)
        else:
            ordinary.append(point)
    return tuple(ordinary), tuple(stiff), completed


def recovery_runs(source: Path) -> tuple[RecoveryRun, ...]:
    ordinary, stiff, _ = classify_source_points(source)
    normal_base = ACCDM_ROOT / "base_models" / "base_model_q5001.json"
    return (
        RecoveryRun(
            role="main",
            campaign_id="accdm_scan_highf_q5001_recovery_10core_v1",
            base_model=normal_base,
            points=ordinary,
            theory_ncpus=10,
            theory_threads=10,
            theory_mem="8gb",
            theory_max_active=120,
        ),
        RecoveryRun(
            role="step-pilot",
            campaign_id="accdm_highf_q5001_20core_pilot_v2",
            base_model=normal_base,
            points=stiff[:1],
            theory_ncpus=20,
            theory_threads=20,
            theory_mem="16gb",
            theory_max_active=1,
        ),
        RecoveryRun(
            role="step-recovery",
            campaign_id="accdm_highf_q5001_20core_recovery_v2",
            base_model=normal_base,
            points=stiff[1:],
            theory_ncpus=20,
            theory_threads=20,
            theory_mem="16gb",
            theory_max_active=max(1, min(60, len(stiff) - 1)),
        ),
    )


def _grid_spec(
    run: RecoveryRun, source: Path, source_config: Mapping[str, Any]
) -> dict[str, Any]:
    fit = dict(source_config.get("fit", {}))
    fit["strategy"] = "standard"
    fit["source_fit_dirs"] = []
    fit["max_attempts"] = 3
    return {
        "schema_version": 1,
        "campaign_id": run.campaign_id,
        "description": (
            f"Frozen {run.role} recovery of incomplete points from {source.name}; "
            "q=5001 exact hierarchy."
        ),
        "points": list(run.points),
        "recovery_source": str(source.resolve()),
        "recovery_role": run.role,
        "theory": {"quality": "production", "max_attempts": 1},
        "fit": fit,
        "cluster": {
            "conda_exe": source_config.get("cluster", {}).get(
                "conda_exe", "/opt/anaconda3/bin/conda"
            ),
            "theory_env": source_config.get("cluster", {}).get(
                "theory_env", "class_accdm"
            ),
            "fit_env": source_config.get("cluster", {}).get(
                "fit_env", "class_accdm"
            ),
            "theory_python": source_config.get("cluster", {}).get(
                "theory_python", ""
            ),
            "fit_python": source_config.get("cluster", {}).get("fit_python", ""),
            "theory_ncpus": run.theory_ncpus,
            "theory_threads": run.theory_threads,
            "theory_mem": run.theory_mem,
            "theory_walltime": "48:00:00",
            "fit_ncpus": 1,
            "fit_mem": "4gb",
            "fit_walltime": "12:00:00",
            "theory_max_active": run.theory_max_active,
            "fit_max_active": 500,
            "max_user_active": 650,
            "poll_seconds": 30,
            "submit_delay_seconds": 0.2,
        },
    }


def prepare_recovery_campaigns(
    source: Path, runs_root: Path
) -> dict[str, Path]:
    """Freeze point selections once and build all non-empty recovery runs."""

    source = source.resolve()
    runs_root = runs_root.resolve()
    source_config = _load_json(source / "campaign.json")
    if source_config.get("model_settings", {}).get("accdm_momentum_bins") != 5001:
        raise ValueError("The source campaign must use 5001 accDM momentum bins.")
    if int(source_config.get("model_settings", {}).get("ncdm_fluid_approximation", -1)) != 3:
        raise ValueError("The source campaign must use the exact NCDM hierarchy.")

    ordinary, stiff, completed = classify_source_points(source)
    print(
        f"source: complete={completed}, ordinary_missing={len(ordinary)}, "
        f"step_size_too_small={len(stiff)}"
    )
    outputs: dict[str, Path] = {}
    for run in recovery_runs(source):
        run_dir = runs_root / run.campaign_id
        spec_path = run_dir / "frozen_recovery_grid.json"
        if spec_path.is_file():
            spec = _load_json(spec_path)
            print(f"reuse frozen selection: {spec_path}")
        else:
            if not run.points:
                print(f"skip empty recovery role: {run.role}")
                continue
            spec = _grid_spec(run, source, source_config)
            _atomic_json(spec_path, spec)
        if spec.get("campaign_id") != run.campaign_id:
            raise ValueError(f"Frozen selection has wrong campaign_id: {spec_path}")
        config = build_campaign(run.base_model, spec_path, run_dir)
        print(
            f"built {run.role}: {run_dir.name}, points={config['point_count']}, "
            f"threads={config['cluster']['theory_threads']}, "
            f"walltime={config['cluster']['theory_walltime']}"
        )
        outputs[run.role] = run_dir
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    args = parser.parse_args(argv)
    prepare_recovery_campaigns(args.source_campaign, args.runs_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
