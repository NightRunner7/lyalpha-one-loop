#!/usr/bin/env python3
"""Replace a bundle's regulated-recursion P13 with closed EdS integrals."""

from __future__ import annotations

import argparse
from pathlib import Path

from lyalpha_pt.theory import load_theory, refine_p13_analytic, save_theory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-q", type=int, default=1001)
    parser.add_argument("--q-min", type=float)
    parser.add_argument("--q-max", type=float)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source = load_theory(args.theory)
    refined = refine_p13_analytic(
        source,
        n_q=args.n_q,
        q_min=args.q_min,
        q_max=args.q_max,
        verbose=not args.quiet,
    )
    save_theory(refined, args.output)
    print(f"saved: {args.output.resolve()}")
    print(f"parent digest: {source.metadata['bundle_digest']}")
    print(f"new digest:    {refined.metadata['bundle_digest']}")


if __name__ == "__main__":
    main()
