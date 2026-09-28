#!/usr/bin/env python3
"""Print and cross-check one theory bundle, fit JSON and cutoff scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lyalpha_pt.theory import CHANNELS, load_theory


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _near_bound(record: dict) -> list[str]:
    return [
        row["parameter"]
        for row in record.get("boundary_diagnostics", [])
        if row.get("near_bound", False)
    ]


def _check_fit_digest(record: dict, digest: str, label: str) -> None:
    stored = record.get("theory_digest")
    if stored != digest:
        raise ValueError(
            f"{label}: theory digest mismatch: JSON={stored!r}, bundle={digest!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--cutoff-scan", type=Path)
    args = parser.parse_args()

    theory = load_theory(args.theory)
    metadata = theory.metadata
    model = metadata["model"]
    numerics = metadata["numerics"]
    derived = metadata.get("derived_cosmology", {})
    digest = metadata["bundle_digest"]
    trusted = theory.k_loop <= float(numerics["k_trust"])

    reconstructed = theory.p_tree[:, :, None] + theory.loop_weight * (
        theory.p22 + theory.p13
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed - theory.channels_one_loop))
    )

    print("=== THEORY BUNDLE ===")
    print(f"file             : {args.theory.resolve()}")
    print(f"model            : {model['name']}")
    print(f"description      : {model.get('description', '')}")
    print(f"digest           : {digest}")
    print(f"redshifts        : {', '.join(f'{z:.1f}' for z in theory.z)}")
    print(
        f"loop k grid      : {theory.k_loop[0]:.6g}--{theory.k_loop[-1]:.6g} "
        f"h/Mpc ({len(theory.k_loop)} points)"
    )
    print(f"k_trust          : {float(numerics['k_trust']):.6g} h/Mpc")
    print(f"internal q range : {numerics['q_min']}--{numerics['q_max']} h/Mpc")
    print(f"P13 method       : {metadata['p13_method']}")
    print(f"loop source      : {metadata['loop_source']}")
    print(f"loop weight      : {theory.loop_weight:.12g}")
    print(f"loop formula     : {metadata['loop_formula']}")
    if derived:
        print(
            "derived          : "
            + ", ".join(f"{name}={value:.8g}" for name, value in derived.items())
        )
    print(f"reconstruction   : max absolute error = {reconstruction_error:.3e}")
    for channel, name in enumerate(CHANNELS):
        values = theory.channels_one_loop[:, trusted, channel]
        print(
            f"trusted {name:10s}: min={np.min(values):.6e}, "
            f"finite={bool(np.all(np.isfinite(values)))}"
        )

    if args.fit:
        fit = _load_json(args.fit)
        print("\n=== FIT ===")
        print(f"file             : {args.fit.resolve()}")
        print(f"parameterization : {fit.get('optimizer_parameterization', 'unknown')}")
        print(f"continuation     : {fit.get('cutoff_continuation_enabled', 'unknown')}")
        records = fit.get("fits", {})
        for mode, record in records.items():
            _check_fit_digest(record, digest, f"fit {mode}")
            near = _near_bound(record)
            print(
                f"{mode:16s}: chi2={record['chi2']:.9f}, "
                f"chi2/dof={record['chi2_per_dof']:.6f}, "
                f"k_uv={record['k_uv_cut_hmpc']:.6g}, "
                f"near_bound={near or 'none'}"
            )
            path = record.get("cutoff_continuation", [])
            if path:
                print(
                    "  cutoff path    : "
                    + " -> ".join(
                        f"{item['k_uv_cut_hmpc']:g} ({item['chi2']:.6f})"
                        for item in path
                    )
                )
            print(
                "  parameters     : "
                + ", ".join(
                    f"{name}={value:.7g}"
                    for name, value in record.get("parameters", {}).items()
                )
            )
        if "delta_chi2_one_loop_minus_linear" in fit:
            print(
                "Delta chi2 1L-lin: "
                f"{fit['delta_chi2_one_loop_minus_linear']:.9f}"
            )

    if args.cutoff_scan:
        scan = _load_json(args.cutoff_scan)
        print("\n=== CUTOFF SCAN ===")
        print(f"file             : {args.cutoff_scan.resolve()}")
        for record in scan.get("cutoff_scan", []):
            _check_fit_digest(record, digest, "cutoff scan")
            near = _near_bound(record)
            print(
                f"k_uv={record['k_uv_cut_hmpc']:6.2f} h/Mpc : "
                f"chi2={record['chi2']:.9f}, near_bound={near or 'none'}"
            )


if __name__ == "__main__":
    main()
