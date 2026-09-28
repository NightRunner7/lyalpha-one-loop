#!/usr/bin/env python3
"""Recompute a bundle's P22 with the symmetric half-domain quadrature."""

from __future__ import annotations

import argparse
from pathlib import Path

from lyalpha_pt.theory import load_theory, refine_p22_symmetric, save_theory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-min", type=float)
    parser.add_argument("--q-max", type=float)
    parser.add_argument("--n-q-low", type=int)
    parser.add_argument("--n-q-mid", type=int)
    parser.add_argument("--n-q-high", type=int)
    parser.add_argument("--n-p", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source = load_theory(args.theory)
    refined = refine_p22_symmetric(
        source,
        q_min=args.q_min,
        q_max=args.q_max,
        n_q_low=args.n_q_low,
        n_q_mid=args.n_q_mid,
        n_q_high=args.n_q_high,
        n_p=args.n_p,
        verbose=not args.quiet,
    )
    save_theory(refined, args.output)
    print(f"saved: {args.output.resolve()}")
    print(f"parent digest: {source.metadata['bundle_digest']}")
    print(f"new digest:    {refined.metadata['bundle_digest']}")


if __name__ == "__main__":
    main()
