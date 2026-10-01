#!/usr/bin/env python3
"""Prepare an isolated accDM Slurm campaign using the active Python environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from campaigns.accdm.build_grid import build_campaign, _atomic_json
from lyalpha_pt.class_runtime import class_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "campaigns/accdm/base_models/base_model_refactor_birth.json"


def prepare(args):
    run_dir = args.run_dir.expanduser().resolve()
    if args.grid_spec and not args.base_model:
        raise ValueError("With --grid-spec, also pass --base-model to select the intended momentum resolution.")
    spec = json.loads(args.grid_spec.read_text()) if args.grid_spec else {
        "campaign_id": run_dir.name,
        "description": "accDM_refactor with the requested CLASS precision; smoke SPT and fit only.",
        "points": [{"log10m_acc": 15, "log10f_acc": -4}],
        "theory": {"quality": "smoke", "max_attempts": 1},
        "fit": {
            "mode": "both", "seeds": [12345], "de_maxiter": 50, "de_popsize": 10,
            "expanded_counterterm": True, "no_cutoff_continuation": True, "max_attempts": 1,
        },
    }
    if args.grid_spec:
        spec["campaign_id"] = run_dir.name
    if args.quality:
        spec.setdefault("theory", {})["quality"] = args.quality
    quality = spec.get("theory", {}).get("quality", "production")
    if not args.grid_spec and quality != "smoke":
        spec["description"] = "Single Eagle accDM anchor; further grid and accuracy validation required."
        spec["fit"].update({"seeds": [12345, 23456, 34567], "de_maxiter": 300, "de_popsize": 20})
    partition = args.partition or ("fast" if quality == "smoke" else "standard")
    walltime = args.walltime or ("01:00:00" if partition == "fast" else "24:00:00")
    if min(args.cpus, args.max_active, args.max_user_active) < 1:
        raise ValueError("CPU and active-job limits must be positive.")
    runtime = class_runtime(source_dir=args.class_source)
    old_path = run_dir / "campaign.json"
    if old_path.exists():
        old_config = json.loads(old_path.read_text())
        old = old_config["cluster"]
        if old.get("scheduler") != "slurm" or old.get("class_source_commit") != runtime["declared_source"]["commit"] or old.get("class_wrapper_sha256") != runtime["classy_sha256"]:
            raise ValueError("Run directory belongs to a different scheduler/CLASS build; choose a new --run-dir.")
        if old_config["theory"]["quality"] != quality:
            raise ValueError("Theory quality changed; choose a new --run-dir instead of reusing smoke results.")
    # Replace site-specific values from the old grid; preserve scientific inputs.
    spec["cluster"] = {
        "scheduler": "slurm", "account": args.account, "partition": partition,
        "theory_python": sys.executable, "fit_python": sys.executable,
        "theory_ncpus": args.cpus, "theory_threads": args.cpus,
        "theory_mem": args.memory, "theory_walltime": walltime,
        "fit_ncpus": 1, "fit_mem": "4gb", "fit_walltime": walltime,
        "theory_max_active": args.max_active, "fit_max_active": args.max_active,
        "max_user_active": args.max_user_active,
        "poll_seconds": 60, "submit_delay_seconds": 0.5,
        "class_source_dir": runtime["declared_source"]["directory"],
        "class_source_commit": runtime["declared_source"]["commit"],
        "class_wrapper_sha256": runtime["classy_sha256"],
    }
    # Keep the exact inputs alongside the resulting manifest.
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "eagle_grid_spec.json"
    _atomic_json(spec_path, spec)
    base_path = (args.base_model or DEFAULT_BASE).resolve()
    config = build_campaign(base_path, spec_path, run_dir)
    _atomic_json(run_dir / "eagle_base_model.json", json.loads(base_path.read_text()))
    _atomic_json(run_dir / "prepared_runtime.json", runtime)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/accdm/eagle_refactor_smoke_v2"))
    parser.add_argument("--class-source", type=Path, required=True)
    parser.add_argument("--account", required=True, help="Your Eagle project/service ID, e.g. pl0503-01.")
    parser.add_argument("--partition", help="Default: fast for smoke, standard otherwise.")
    parser.add_argument("--walltime", help="HH:MM:SS; default 1 h for fast or 24 h otherwise.")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory", default="8gb", help="Per theory job, e.g. 8gb.")
    parser.add_argument("--max-active", type=int, default=4, help="Local controller cap per stage, not a site limit.")
    parser.add_argument("--max-user-active", type=int, default=100, help="Cap counting all jobs of this Slurm user.")
    parser.add_argument("--grid-spec", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--quality", choices=("smoke", "production", "precision"))
    args = parser.parse_args()
    config = prepare(args)
    print(f"Prepared {config['campaign_id']}: {config['point_count']} point(s)")
    print(f"Campaign: {args.run_dir.resolve()}")
    print(f"Python: {sys.executable}")
    print(f"CLASS commit: {config['cluster']['class_source_commit']}")
    print(f"Model profile: {config['base_model_name']}")
    print("CLASS sampling: " + json.dumps(config["model_settings"], sort_keys=True))
    print("--quality changes SPT resolution and fit defaults, not the CLASS model profile.")
    print("No jobs submitted. Next: python -m cluster.campaign_manager submit-theory --campaign "
          f"{args.run_dir} --dry-run --offline --max-points 1")


if __name__ == "__main__":
    main()
