#!/usr/bin/env python3
"""Compare raw loop channels from two bundles on the trusted k range."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from lyalpha_pt.theory import CHANNELS, load_theory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--k-max", type=float, default=2.0)
    args = parser.parse_args()

    first = load_theory(args.first)
    second = load_theory(args.second)
    if not np.allclose(first.z, second.z, rtol=0, atol=1e-9):
        raise ValueError("The bundles use different redshift grids.")

    k_min = max(float(first.k_loop[0]), float(second.k_loop[0]))
    k_max = min(args.k_max, float(first.k_loop[-1]), float(second.k_loop[-1]))
    k = np.geomspace(k_min, k_max, 500)
    print(f"comparison range: {k_min:.6g}--{k_max:.6g} h/Mpc")
    print("relative difference = (second-first)/max(|first|, 1e-12*tree)")

    for iz, redshift in enumerate(first.z):
        first_tree = InterpolatedUnivariateSpline(
            np.log(first.k_loop), first.p_tree[iz], k=3
        )(np.log(k))
        for channel, name in enumerate(CHANNELS):
            values = []
            for bundle in (first, second):
                spline = InterpolatedUnivariateSpline(
                    np.log(bundle.k_loop),
                    bundle.channels_one_loop[iz, :, channel],
                    k=3,
                )
                values.append(spline(np.log(k)))
            denominator = np.maximum(np.abs(values[0]), 1e-12 * np.abs(first_tree))
            relative = (values[1] - values[0]) / denominator
            print(
                f"z={redshift:.1f} {name:10s} "
                f"max_abs={np.max(np.abs(relative)):.6e} "
                f"rms={np.sqrt(np.mean(relative**2)):.6e}"
            )


if __name__ == "__main__":
    main()
