#!/usr/bin/env python3
"""Repeat local fits for several technical P1D UV cutoffs."""

from __future__ import annotations

import argparse
from pathlib import Path

from lyalpha_pt.data import load_dr12
from lyalpha_pt.fit import EffectiveModelFit, write_fit_result
from lyalpha_pt.theory import load_theory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[10.0, 15.0, 20.0])
    parser.add_argument("--mode", choices=("linear", "one_loop"), default="one_loop")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567])
    parser.add_argument("--de-maxiter", type=int, default=300)
    parser.add_argument("--de-popsize", type=int, default=20)
    parser.add_argument("--expanded-counterterm", action="store_true")
    parser.add_argument("--full-six-dimensional", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load_dr12(args.data_dir)
    theory = load_theory(args.theory)
    records = []
    continuation_thetas = []
    for cutoff in args.cutoffs:
        fitter = EffectiveModelFit(
            data,
            theory,
            mode=args.mode,
            k_uv_cut=cutoff,
            covariance_mode="paper_diag",
        )
        result = fitter.fit_staged(
            seeds=args.seeds,
            expand_counterterm=args.expanded_counterterm,
            de_maxiter=args.de_maxiter,
            de_popsize=args.de_popsize,
            profile_amplitudes=not args.full_six_dimensional,
            initial_thetas=continuation_thetas,
        )
        record = fitter.result_record(result)
        records.append(record)
        continuation_thetas = [result.x]
        print(f"k_uv={cutoff:5.1f} h/Mpc: chi2={record['chi2']:.6f}")
    write_fit_result(
        args.output,
        {
            "theory_file": str(args.theory.resolve()),
            "mode": args.mode,
            "optimizer_parameterization": (
                "full_six_dimensional"
                if args.full_six_dimensional
                else "four_dimensional_with_two_profiled_amplitudes"
            ),
            "cutoff_scan": records,
        },
    )


if __name__ == "__main__":
    main()
