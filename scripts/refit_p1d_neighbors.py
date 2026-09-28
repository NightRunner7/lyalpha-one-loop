#!/usr/bin/env python3
"""Refit one completed campaign point from its own and neighbouring fits."""

from __future__ import annotations

import argparse
from pathlib import Path

from lyalpha_pt.campaign_refit import refit_campaign_point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--point-id", required=True)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fit-dir", type=Path, action="append", required=True)
    parser.add_argument("--mode", choices=("linear", "one_loop", "both"), default="one_loop")
    parser.add_argument("--k-uv-cut", type=float, default=20.0)
    parser.add_argument("--covariance", default="paper_diag")
    parser.add_argument("--expanded-counterterm", action="store_true")
    parser.add_argument("--full-six-dimensional", action="store_true")
    args = parser.parse_args()

    modes = ("linear", "one_loop") if args.mode == "both" else (args.mode,)
    result = refit_campaign_point(
        campaign_dir=args.campaign_dir,
        point_id=args.point_id,
        theory_path=args.theory,
        data_dir=args.data_dir,
        output_path=args.output,
        source_fit_directories=args.source_fit_dir,
        modes=modes,
        k_uv_cut=args.k_uv_cut,
        covariance=args.covariance,
        expanded_counterterm=args.expanded_counterterm,
        full_six_dimensional=args.full_six_dimensional,
    )
    for mode, record in result["fits"].items():
        selected = record["neighbour_refit_selected_source"]
        print(
            f"{mode:8s}: chi2={record['chi2']:.6f}, "
            f"source={Path(selected['source_directory']).name}:"
            f"{selected['source_point_id']}:{selected['candidate_kind']}"
        )
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
