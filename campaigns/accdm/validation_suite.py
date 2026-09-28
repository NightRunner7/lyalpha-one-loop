#!/usr/bin/env python3
"""Build and control the parallel accDM validation and q-convergence suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from campaigns.accdm.build_grid import build_campaign


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCDM_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SuiteCampaign:
    name: str
    base_model: Path
    grid_spec: Path


SUITE_CAMPAIGNS = (
    SuiteCampaign(
        "accdm_validation_mass11_v1",
        ACCDM_ROOT / "base_model.json",
        ACCDM_ROOT / "grid_specs" / "validation_mass11.json",
    ),
    SuiteCampaign(
        "accdm_qconvergence_f0p3_q1001_v1",
        ACCDM_ROOT / "base_model.json",
        ACCDM_ROOT / "grid_specs" / "qconvergence_f0p3_q1001.json",
    ),
    SuiteCampaign(
        "accdm_qconvergence_joint_q2501_v1",
        ACCDM_ROOT / "base_models" / "base_model_q2501.json",
        ACCDM_ROOT / "grid_specs" / "qconvergence_joint_q2501.json",
    ),
    SuiteCampaign(
        "accdm_qconvergence_joint_q5001_v1",
        ACCDM_ROOT / "base_models" / "base_model_q5001.json",
        ACCDM_ROOT / "grid_specs" / "qconvergence_joint_q5001.json",
    ),
)


def campaign_dir(runs_root: Path, campaign: SuiteCampaign) -> Path:
    return runs_root.resolve() / campaign.name


def build_suite(runs_root: Path) -> tuple[Path, ...]:
    """Build every immutable campaign and return their run directories."""

    outputs: list[Path] = []
    for campaign in SUITE_CAMPAIGNS:
        run_dir = campaign_dir(runs_root, campaign)
        config = build_campaign(campaign.base_model, campaign.grid_spec, run_dir)
        print(
            f"built {campaign.name}: {config['point_count']} point(s), "
            f"q={config['model_settings']['accdm_momentum_bins']}"
        )
        outputs.append(run_dir)
    return tuple(outputs)


def manager_command(
    project_root: Path,
    command: str,
    run_dir: Path,
    extra: Sequence[str] = (),
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(project_root / "cluster" / "campaign_manager.py"),
        command,
        "--campaign",
        str(run_dir),
        "--project-dir",
        str(project_root),
        *extra,
    ]


def require_built(runs_root: Path) -> tuple[Path, ...]:
    run_dirs = tuple(campaign_dir(runs_root, item) for item in SUITE_CAMPAIGNS)
    missing = [str(path) for path in run_dirs if not (path / "campaign.json").is_file()]
    if missing:
        raise FileNotFoundError(
            "Build the validation suite first. Missing campaign.json in: "
            + ", ".join(missing)
        )
    return run_dirs


def run_for_each(
    project_root: Path,
    runs_root: Path,
    command: str,
    extra: Sequence[str] = (),
) -> int:
    return_code = 0
    for run_dir in require_built(runs_root):
        print(f"\n=== {run_dir.name} ===", flush=True)
        result = subprocess.run(
            manager_command(project_root, command, run_dir, extra),
            cwd=project_root,
            check=False,
        )
        return_code = max(return_code, result.returncode)
    return return_code


def watch_parallel(
    project_root: Path,
    runs_root: Path,
    stages: Sequence[str],
) -> int:
    """Run one campaign controller per suite member at the same time."""

    processes: list[tuple[Path, subprocess.Popen[str], object]] = []
    for run_dir in require_built(runs_root):
        stop_file = run_dir / "controller" / "STOP"
        if stop_file.exists():
            raise RuntimeError(
                f"Stop request exists for {run_dir.name}. Run the suite resume command first."
            )
        log_path = run_dir / "controller" / "watch.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = manager_command(
            project_root,
            "watch",
            run_dir,
            ("--stages", *stages, "--exit-when-complete"),
        )
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((run_dir, process, handle))
        print(
            f"started {run_dir.name}: pid={process.pid}, log={log_path}",
            flush=True,
        )

    print("All validation controllers are running in parallel.", flush=True)
    print("Interrupting this process stops controllers but does not cancel PBS jobs.", flush=True)
    exit_code = 0
    try:
        remaining = {process.pid for _, process, _ in processes}
        while remaining:
            for run_dir, process, _ in processes:
                if process.pid not in remaining:
                    continue
                code = process.poll()
                if code is None:
                    continue
                remaining.remove(process.pid)
                exit_code = max(exit_code, code)
                print(
                    f"controller finished: {run_dir.name}, exit_code={code}",
                    flush=True,
                )
            if remaining:
                time.sleep(5.0)
    except KeyboardInterrupt:
        print("Stopping validation controllers; submitted PBS jobs are left untouched.")
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            if process.poll() is None:
                process.wait(timeout=30)
        exit_code = 130
    finally:
        for _, _, handle in processes:
            handle.close()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "accdm",
        help="Parent directory for the four immutable validation campaigns.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing cluster/, scripts/, and data/.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("status")
    subparsers.add_parser("dry-run")
    subparsers.add_parser("stop")
    subparsers.add_parser("resume")
    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument(
        "--stages",
        nargs="+",
        choices=("theory", "fit"),
        default=("theory", "fit"),
    )
    args = parser.parse_args()

    project_root = args.project_dir.resolve()
    runs_root = args.runs_root.resolve()
    if args.command == "build":
        build_suite(runs_root)
        return 0
    if args.command == "status":
        return run_for_each(project_root, runs_root, "status")
    if args.command == "dry-run":
        return run_for_each(
            project_root,
            runs_root,
            "submit-theory",
            ("--dry-run", "--offline", "--max-points", "1"),
        )
    if args.command == "stop":
        return run_for_each(project_root, runs_root, "stop")
    if args.command == "resume":
        return run_for_each(project_root, runs_root, "resume")
    if args.command == "watch":
        return watch_parallel(project_root, runs_root, args.stages)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
