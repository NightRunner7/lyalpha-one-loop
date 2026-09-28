#!/usr/bin/env python3
"""Cluster entry point: CLASS + direct one-loop SPT -> one theory file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lyalpha_pt.models import ModelSpec, available_presets, get_preset
from lyalpha_pt.theory import LoopNumerics, generate_theory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--preset", choices=available_presets())
    model_group.add_argument(
        "--model-json",
        type=Path,
        help="JSON file containing one serialized ModelSpec",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality", choices=("smoke", "production", "precision"), default="production"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--overwrite-checkpoints", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset:
        model = get_preset(args.preset)
    else:
        model = ModelSpec.from_dict(json.loads(args.model_json.read_text()))
    numerics = LoopNumerics.for_quality(args.quality)
    bundle = generate_theory(
        model,
        args.output,
        numerics=numerics,
        checkpoint_dir=args.checkpoint_dir,
        overwrite_checkpoints=args.overwrite_checkpoints,
        verbose=not args.quiet,
    )
    print(f"saved: {args.output.resolve()}")
    print(f"model: {model.name}")
    print(f"digest: {bundle.metadata['bundle_digest']}")
    print(f"loop weight: {bundle.loop_weight:.12g}")
    energy_budget = bundle.metadata.get("accdm_energy_budget")
    if energy_budget is not None:
        print(
            "accDM energy-budget ratio "
            "eta*m/[rho_crit(a_t)*H(a_t)^-3]: "
            f"{energy_budget['ratio']:.6e} "
            f"(log10={energy_budget['log10_ratio']:.3f})"
        )


if __name__ == "__main__":
    main()
