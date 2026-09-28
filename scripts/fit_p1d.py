#!/usr/bin/env python3
"""Local entry point: saved theory + public data -> six-parameter P1D fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lyalpha_pt.data import load_dr12
from lyalpha_pt.fit import EffectiveModelFit, write_fit_result
from lyalpha_pt.theory import load_theory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=("linear", "one_loop", "both"), default="both")
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567])
    parser.add_argument(
        "--expanded-counterterm",
        action="store_true",
        help=(
            "After the base-box search, continue all candidates into the wider "
            "counterterm box. The wide box is never used as the first search stage."
        ),
    )
    parser.add_argument(
        "--full-six-dimensional",
        action="store_true",
        help=(
            "Disable exact profiling of the two linear amplitude combinations "
            "and run the legacy six-dimensional global search."
        ),
    )
    parser.add_argument("--de-maxiter", type=int, default=300)
    parser.add_argument("--de-popsize", type=int, default=20)
    parser.add_argument(
        "--no-cutoff-continuation",
        action="store_true",
        help=(
            "Fit only the requested cutoff. By default a target above 10 h/Mpc "
            "is reached through the deterministic ladder 10 -> 15 -> target."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dr12(args.data_dir)
    theory = load_theory(args.theory)
    modes = ("linear", "one_loop") if args.mode == "both" else (args.mode,)
    records = {}

    for mode in modes:
        if args.no_cutoff_continuation or args.k_uv_cut <= 10.0:
            cutoff_path = [args.k_uv_cut]
        else:
            cutoff_path = [10.0]
            if args.k_uv_cut > 15.0:
                cutoff_path.append(15.0)
            cutoff_path.append(args.k_uv_cut)
            cutoff_path = list(dict.fromkeys(cutoff_path))

        continuation = []
        result = None
        fitter = None
        for index, cutoff in enumerate(cutoff_path):
            fitter = EffectiveModelFit(
                data,
                theory,
                mode=mode,
                k_uv_cut=cutoff,
                covariance_mode=args.covariance,
            )
            if index == 0:
                result = fitter.fit_staged(
                    seeds=args.seeds,
                    expand_counterterm=args.expanded_counterterm,
                    de_maxiter=args.de_maxiter,
                    de_popsize=args.de_popsize,
                    profile_amplitudes=not args.full_six_dimensional,
                )
            else:
                result = fitter.continue_staged(
                    result.x,
                    expand_counterterm=args.expanded_counterterm,
                    profile_amplitudes=not args.full_six_dimensional,
                )
            continuation.append(
                {"k_uv_cut_hmpc": float(cutoff), "chi2": float(fitter.chi2(result.x))}
            )

        assert fitter is not None and result is not None
        record = fitter.result_record(result)
        record["cutoff_continuation"] = continuation
        records[mode] = record
        print(
            f"{mode:8s}: chi2={record['chi2']:.6f}, "
            f"chi2/dof={record['chi2_per_dof']:.6f}, "
            f"near_bound={any(row['near_bound'] for row in record['boundary_diagnostics'])}"
        )

    output = {
        "theory_file": str(args.theory.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "optimizer_parameterization": (
            "full_six_dimensional"
            if args.full_six_dimensional
            else "four_dimensional_with_two_profiled_amplitudes"
        ),
        "cutoff_continuation_enabled": not args.no_cutoff_continuation,
        "fits": records,
    }
    if set(records) == {"linear", "one_loop"}:
        output["delta_chi2_one_loop_minus_linear"] = (
            records["one_loop"]["chi2"] - records["linear"]["chi2"]
        )
        print(
            "Delta chi2 (one_loop-linear) = "
            f"{output['delta_chi2_one_loop_minus_linear']:.6f}"
        )
    write_fit_result(args.output, output)
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
