#!/usr/bin/env python3
"""Cross-start two saved P1D fits on two saved theory bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lyalpha_pt.data import load_dr12
from lyalpha_pt.fit import EffectiveModelFit, PARAMETER_NAMES
from lyalpha_pt.theory import load_theory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--left-theory", type=Path, required=True)
    parser.add_argument("--left-fit", type=Path, required=True)
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--right-theory", type=Path, required=True)
    parser.add_argument("--right-fit", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=("linear", "one_loop"), default="one_loop")
    parser.add_argument("--k-uv-cut", type=float, default=20.0)
    parser.add_argument(
        "--covariance",
        choices=(
            "paper_diag",
            "stat_diag",
            "stat_corr",
            "total_corr",
            "outer_systematics",
        ),
        default="paper_diag",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_fit_record(path: Path, mode: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        record = payload["fits"][mode]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} does not contain fits.{mode}.") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path} fits.{mode} is not an object.")
    return record


def theta_from_record(record: Mapping[str, Any]) -> np.ndarray:
    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Fit record has no parameter mapping.")
    try:
        return np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=float)
    except KeyError as error:
        raise ValueError(f"Fit record is missing parameter {error.args[0]!r}.") from error


def transfer_counterterm_amplitude(
    theta: Sequence[float], source_i0_scale: float, target_i0_scale: float
) -> np.ndarray:
    """Preserve the dimensional counterterm amplitude in an imported start."""

    transferred = np.asarray(theta, dtype=float).copy()
    if transferred.shape != (6,):
        raise ValueError("theta must contain six parameters")
    if not np.isfinite(source_i0_scale) or not np.isfinite(target_i0_scale):
        raise ValueError("I0 scales must be finite.")
    if target_i0_scale == 0.0:
        raise ValueError("Target I0 scale cannot be zero.")
    alpha_ct_index = PARAMETER_NAMES.index("alpha_ct")
    transferred[alpha_ct_index] *= source_i0_scale / target_i0_scale
    return transferred


def minimum_bound_distance(result) -> tuple[str, float]:
    row = min(
        result.boundary_diagnostics,
        key=lambda item: item["relative_distance_to_nearest_bound"],
    )
    return str(row["parameter"]), float(row["relative_distance_to_nearest_bound"])


def refine_target(
    *,
    target_label: str,
    target_record: Mapping[str, Any],
    target_fitter: EffectiveModelFit,
    source_label: str,
    source_record: Mapping[str, Any],
    source_fitter: EffectiveModelFit,
) -> dict[str, Any]:
    target_theta = theta_from_record(target_record)
    source_theta = theta_from_record(source_record)
    imported_preserved = transfer_counterterm_amplitude(
        source_theta,
        source_fitter.i0_scale,
        target_fitter.i0_scale,
    )
    starts = {
        "stored_target": target_theta,
        f"raw_from_{source_label}": source_theta,
        f"counterterm_preserved_from_{source_label}": imported_preserved,
    }

    attempts = []
    for start_label, theta in starts.items():
        before = float(target_fitter.chi2(theta))
        result = target_fitter.continue_staged(
            theta,
            expand_counterterm=True,
            profile_amplitudes=True,
        )
        after = float(target_fitter.chi2(result.x))
        bound_parameter, bound_distance = minimum_bound_distance(result)
        attempts.append(
            {
                "start": start_label,
                "chi2_before": before,
                "chi2_after": after,
                "improvement": before - after,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "nearest_bound_parameter": bound_parameter,
                "nearest_bound_distance": bound_distance,
                "parameters": {
                    name: float(value)
                    for name, value in zip(PARAMETER_NAMES, result.x)
                },
            }
        )

    best = min(attempts, key=lambda item: item["chi2_after"])
    stored_chi2 = float(target_record["chi2"])
    evaluated_stored_chi2 = float(target_fitter.chi2(target_theta))
    return {
        "target": target_label,
        "source": source_label,
        "stored_chi2": stored_chi2,
        "evaluated_stored_chi2": evaluated_stored_chi2,
        "stored_chi2_reproduction_error": evaluated_stored_chi2 - stored_chi2,
        "best_refined_chi2": float(best["chi2_after"]),
        "best_improvement_from_stored": stored_chi2 - float(best["chi2_after"]),
        "best_start": str(best["start"]),
        "attempts": attempts,
    }


def print_target(record: Mapping[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"Target theory: {record['target']}; imported source: {record['source']}")
    print(f"stored chi2            = {record['stored_chi2']:.9f}")
    print(f"evaluated stored chi2  = {record['evaluated_stored_chi2']:.9f}")
    print(f"best refined chi2      = {record['best_refined_chi2']:.9f}")
    print(
        "improvement vs stored  = "
        f"{record['best_improvement_from_stored']:+.9f}"
    )
    print(f"best start             = {record['best_start']}")
    print("attempts:")
    for attempt in record["attempts"]:
        print(
            f"  {attempt['start']:<42s} "
            f"before={attempt['chi2_before']:.9f} "
            f"after={attempt['chi2_after']:.9f} "
            f"gain={attempt['improvement']:+.9f} "
            f"bound={attempt['nearest_bound_parameter']} "
            f"({100.0 * attempt['nearest_bound_distance']:.2f}%)"
        )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    dataset = load_dr12(args.data_dir)
    left_fitter = EffectiveModelFit(
        dataset,
        load_theory(args.left_theory),
        mode=args.mode,
        k_uv_cut=args.k_uv_cut,
        covariance_mode=args.covariance,
    )
    right_fitter = EffectiveModelFit(
        dataset,
        load_theory(args.right_theory),
        mode=args.mode,
        k_uv_cut=args.k_uv_cut,
        covariance_mode=args.covariance,
    )
    left_record = load_fit_record(args.left_fit, args.mode)
    right_record = load_fit_record(args.right_fit, args.mode)

    left_result = refine_target(
        target_label=args.left_label,
        target_record=left_record,
        target_fitter=left_fitter,
        source_label=args.right_label,
        source_record=right_record,
        source_fitter=right_fitter,
    )
    right_result = refine_target(
        target_label=args.right_label,
        target_record=right_record,
        target_fitter=right_fitter,
        source_label=args.left_label,
        source_record=left_record,
        source_fitter=left_fitter,
    )
    payload = {
        "mode": args.mode,
        "k_uv_cut_hmpc": float(args.k_uv_cut),
        "covariance": args.covariance,
        "left_theory": str(args.left_theory.resolve()),
        "left_fit": str(args.left_fit.resolve()),
        "right_theory": str(args.right_theory.resolve()),
        "right_fit": str(args.right_fit.resolve()),
        "targets": [left_result, right_result],
    }
    print_target(left_result)
    print_target(right_result)
    if args.output is not None:
        write_json(args.output, payload)
        print(f"\nsaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
