#!/usr/bin/env python3
"""Validate raw one-loop SPT channels without the P1D likelihood or fit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from lyalpha_pt.spt import (
    p13_channels_analytic,
    p13_channels_richardson,
    p22_channels,
    p22_channels_full_qp_reference,
)
from lyalpha_pt.theory import CHANNELS, load_theory


DEFAULT_K = (0.01, 0.03, 0.1, 0.3, 1.0, 2.0)


def _spectrum_callable(k: np.ndarray, power: np.ndarray):
    spline = InterpolatedUnivariateSpline(np.log(k), np.log(power), k=3)

    def evaluate(values):
        return np.exp(spline(np.log(np.asarray(values, dtype=float))))

    return evaluate


def _value_spline(k: np.ndarray, values: np.ndarray):
    return InterpolatedUnivariateSpline(np.log(k), values, k=3)


def _fastpt_channels(bundle, iz: int, n_grid: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Independent FFTLog result for dd, dtheta and thetatheta loop sums.

    FAST-PT regularizes P22 and P13 separately before adding them.  Therefore
    only the sum is compared to our unregularized split.
    """

    import fastpt
    from fastpt.IA.IA_ct import (
        P_22F_reg,
        P_22G_reg,
        P_IA_13F,
        P_IA_13G,
    )
    from fastpt.utils.J_k import J_k

    n_grid = int(n_grid)
    if n_grid < 1024 or n_grid % 2:
        raise ValueError("FAST-PT grid must be even and contain at least 1024 points.")
    q_min = float(bundle.metadata["numerics"]["q_min"])
    q_max = float(bundle.metadata["numerics"]["q_max"])
    q_max = min(q_max, float(bundle.k_input[-1]) - float(bundle.k_loop[-1]))
    k = np.geomspace(q_min, q_max, n_grid)
    linear = _spectrum_callable(bundle.k_input, bundle.p_loop_input[iz])(k)
    n_pad = n_grid // 2

    p22_dd = P_22F_reg(k, linear, None, None, n_pad)
    p22_dtheta = P_22G_reg(k, linear, None, None, n_pad)

    parameter_matrix = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 4, 0],
            [2, -2, 2, 0],
            [1, -1, 1, 0],
            [1, -1, 3, 0],
            [2, -2, 0, 1],
        ]
    )
    _, matrix = J_k(
        k,
        linear,
        parameter_matrix,
        P_window=None,
        C_window=None,
        n_pad=n_pad,
    )
    theta_coefficients = np.array(
        [
            2 * 851 / 1470,
            2 * 871 / 1029,
            2 * 128 / 1715,
            2 / 3,
            2 * 54 / 35,
            2 * 16 / 35,
            1 / 3,
        ]
    )
    p22_thetatheta = np.sum(theta_coefficients[:, None] * matrix, axis=0)

    p13_dd = P_IA_13F(k, linear)
    p13_thetatheta = P_IA_13G(k, linear)
    p13_dtheta = 0.5 * (p13_dd + p13_thetatheta)
    loop = np.column_stack(
        [
            p22_dd + p13_dd,
            p22_dtheta + p13_dtheta,
            p22_thetatheta + p13_thetatheta,
        ]
    )
    return k, loop, {"package": "fast-pt", "version": fastpt.__version__}


def _compare_fastpt(bundle, iz: int, n_grid: int) -> tuple[dict, np.ndarray, np.ndarray]:
    k, reference, metadata = _fastpt_channels(bundle, iz, n_grid)
    trust = float(bundle.metadata["numerics"]["k_trust"])
    mask = (k >= bundle.k_loop[0]) & (k <= trust)
    k_eval = k[mask]
    ours = np.column_stack(
        [
            _value_spline(bundle.k_loop, bundle.p22[iz, :, channel] + bundle.p13[iz, :, channel])(
                np.log(k_eval)
            )
            for channel in range(3)
        ]
    )
    tree = _spectrum_callable(bundle.k_input, bundle.p_loop_input[iz])(k_eval)
    residual = (reference[mask] - ours) / tree[:, None]
    summary = {
        **metadata,
        "k_min_hmpc": float(k_eval[0]),
        "k_max_hmpc": float(k_eval[-1]),
        "n_k": int(len(k_eval)),
        "max_abs_delta_loop_over_tree": {
            channel: float(np.max(np.abs(residual[:, index])))
            for index, channel in enumerate(CHANNELS)
        },
        "rms_delta_loop_over_tree": {
            channel: float(np.sqrt(np.mean(residual[:, index] ** 2)))
            for index, channel in enumerate(CHANNELS)
        },
        "p99_abs_delta_loop_over_tree": {
            channel: float(np.percentile(np.abs(residual[:, index]), 99))
            for index, channel in enumerate(CHANNELS)
        },
    }
    return summary, k_eval, residual


def _compare_bundles(reference, other, redshift: float) -> dict:
    iz_ref = int(np.argmin(np.abs(reference.z - redshift)))
    iz_other = int(np.argmin(np.abs(other.z - redshift)))
    if not np.isclose(reference.z[iz_ref], other.z[iz_other], atol=1e-9, rtol=0):
        raise ValueError("Compared bundles do not contain the same requested redshift.")
    k_min = max(float(reference.k_loop[0]), float(other.k_loop[0]))
    k_max = min(
        float(reference.metadata["numerics"]["k_trust"]),
        float(other.metadata["numerics"]["k_trust"]),
        float(reference.k_loop[-1]),
        float(other.k_loop[-1]),
    )
    k = np.geomspace(k_min, k_max, 500)
    tree = _value_spline(reference.k_loop, reference.p_tree[iz_ref])(np.log(k))
    result = {}
    for component_name, reference_component, other_component in (
        ("p22", reference.p22, other.p22),
        ("p13", reference.p13, other.p13),
        ("loop_sum", reference.p22 + reference.p13, other.p22 + other.p13),
        ("total", reference.channels_one_loop, other.channels_one_loop),
    ):
        delta = np.column_stack(
            [
                _value_spline(other.k_loop, other_component[iz_other, :, channel])(
                    np.log(k)
                )
                - _value_spline(
                    reference.k_loop, reference_component[iz_ref, :, channel]
                )(np.log(k))
                for channel in range(3)
            ]
        )
        result[component_name] = {
            "max_abs_delta_over_reference_tree": {
                channel: float(np.max(np.abs(delta[:, index] / tree)))
                for index, channel in enumerate(CHANNELS)
            },
            "rms_delta_over_reference_tree": {
                channel: float(np.sqrt(np.mean((delta[:, index] / tree) ** 2)))
                for index, channel in enumerate(CHANNELS)
            },
        }
    return result


def _make_plots(output_dir: Path, bundle, iz: int, fastpt_data) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = bundle.k_loop <= float(bundle.metadata["numerics"]["k_trust"])
    k = bundle.k_loop[mask]
    tree = bundle.p_tree[iz, mask]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for channel, axis in enumerate(axes):
        axis.plot(k, bundle.p22[iz, mask, channel] / tree, label=r"$P_{22}/P_{\rm lin}$")
        axis.plot(k, bundle.p13[iz, mask, channel] / tree, label=r"$P_{13}/P_{\rm lin}$")
        axis.plot(
            k,
            (bundle.p22[iz, mask, channel] + bundle.p13[iz, mask, channel]) / tree,
            label=r"$(P_{22}+P_{13})/P_{\rm lin}$",
            linewidth=2,
        )
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(CHANNELS[channel])
        axis.set_xlabel(r"$k\,[h/{\rm Mpc}]$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("raw correction / linear spectrum")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        output_dir / "raw_loop_channels_ktrust.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    full_tree = bundle.p_tree[iz]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for channel, name in enumerate(CHANNELS):
        axis.plot(
            bundle.k_loop,
            bundle.channels_one_loop[iz, :, channel] / full_tree,
            label=name,
        )
    axis.axvline(
        float(bundle.metadata["numerics"]["k_trust"]),
        color="black",
        linestyle="--",
        linewidth=1,
        label=r"$k_{\rm trust}$",
    )
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xlabel(r"$k\,[h/{\rm Mpc}]$")
    axis.set_ylabel(r"$P_{XY}^{\rm 1-loop}/P_{\rm lin}$")
    axis.set_title("Full one-loop channels on the stored grid")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "raw_total_channel_ratios.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    if fastpt_data is not None:
        k_fastpt, residual = fastpt_data
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for channel, name in enumerate(CHANNELS):
            axis.plot(k_fastpt, residual[:, channel], label=name)
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel(r"$k\,[h/{\rm Mpc}]$")
        axis.set_ylabel(r"$(P_{\rm loop}^{\rm FASTPT}-P_{\rm loop})/P_{\rm lin}$")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "fastpt_residuals.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--compare-theory", type=Path, nargs="*", default=[])
    parser.add_argument("--redshift", type=float, default=3.0)
    parser.add_argument("--k-values", type=float, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--p22-reference-nq", type=int, default=1601)
    parser.add_argument("--p22-reference-np", type=int, default=320)
    parser.add_argument("--p13-recursive-nq", type=int, default=180)
    parser.add_argument("--p13-recursive-nmu", type=int, default=40)
    parser.add_argument("--p13-recursive-epsilon", type=float, default=1e-4)
    parser.add_argument("--fastpt-n", type=int, default=4096)
    parser.add_argument("--skip-fastpt", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_theory(args.theory)
    iz = int(np.argmin(np.abs(bundle.z - args.redshift)))
    if not np.isclose(bundle.z[iz], args.redshift, rtol=0, atol=1e-9):
        raise ValueError(f"z={args.redshift} is absent from the bundle.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    numerics = bundle.metadata["numerics"]
    p_linear = _spectrum_callable(bundle.k_input, bundle.p_loop_input[iz])
    selected_indices = sorted(
        {int(np.argmin(np.abs(bundle.k_loop - value))) for value in args.k_values}
    )
    point_rows = []
    for index in selected_indices:
        k = float(bundle.k_loop[index])
        p22_recomputed = p22_channels(
            k,
            p_linear,
            q_min=float(numerics["q_min"]),
            q_max=float(numerics["q_max"]),
            n_q_low=int(numerics["n_q_low"]),
            n_q_mid=int(numerics["n_q_mid"]),
            n_q_high=int(numerics["n_q_high"]),
            n_p=int(numerics["n_p"]),
            middle_width=float(numerics["middle_width"]),
        )
        p22_reference = p22_channels_full_qp_reference(
            k,
            p_linear,
            q_min=float(numerics["q_min"]),
            q_max=float(numerics["q_max"]),
            n_q_per_region=args.p22_reference_nq,
            n_p=args.p22_reference_np,
        )
        p13_analytic, p13_resolution = p13_channels_analytic(
            k,
            p_linear,
            q_min=float(numerics["q_min"]),
            q_max=float(numerics["q_max"]),
            n_q=int(numerics["p13_n_q"]),
        )
        p13_recursive, p13_recursive_error = p13_channels_richardson(
            k,
            p_linear,
            q_min=float(numerics["q_min"]),
            q_max=float(numerics["q_max"]),
            n_q=args.p13_recursive_nq,
            n_mu=args.p13_recursive_nmu,
            epsilon_relative=args.p13_recursive_epsilon,
        )
        tree = float(bundle.p_tree[iz, index])
        for channel, name in enumerate(CHANNELS):
            point_rows.append(
                {
                    "z": float(bundle.z[iz]),
                    "k_hmpc": k,
                    "channel": name,
                    "tree": tree,
                    "bundle_p22": float(bundle.p22[iz, index, channel]),
                    "recomputed_p22": float(p22_recomputed[channel]),
                    "full_qp_reference_p22": float(p22_reference[channel]),
                    "p22_recomputed_minus_reference_over_tree": float(
                        (p22_recomputed[channel] - p22_reference[channel]) / tree
                    ),
                    "bundle_p13": float(bundle.p13[iz, index, channel]),
                    "analytic_p13": float(p13_analytic[channel]),
                    "recursive_p13": float(p13_recursive[channel]),
                    "p13_analytic_minus_recursive_over_tree": float(
                        (p13_analytic[channel] - p13_recursive[channel]) / tree
                    ),
                    "analytic_p13_resolution_error": float(p13_resolution[channel]),
                    "recursive_p13_richardson_error": float(
                        p13_recursive_error[channel]
                    ),
                    "loop_sum_over_tree": float(
                        (bundle.p22[iz, index, channel] + bundle.p13[iz, index, channel])
                        / tree
                    ),
                    "total_over_tree": float(
                        bundle.channels_one_loop[iz, index, channel] / tree
                    ),
                }
            )

    csv_path = args.output_dir / "raw_spt_pointwise.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)

    reconstructed = bundle.p_tree[:, :, None] + bundle.loop_weight * (
        bundle.p22 + bundle.p13
    )
    cross_identity = bundle.p13[:, :, 1] - 0.5 * (
        bundle.p13[:, :, 0] + bundle.p13[:, :, 2]
    )
    cauchy_margin = bundle.p22[:, :, 0] * bundle.p22[:, :, 2] - bundle.p22[:, :, 1] ** 2
    low_mask = (bundle.k_loop >= 1e-3) & (bundle.k_loop <= 1e-2)
    ir_slopes = {}
    for channel, name in enumerate(CHANNELS):
        ratio = np.abs(
            (bundle.p22[iz, low_mask, channel] + bundle.p13[iz, low_mask, channel])
            / bundle.p_tree[iz, low_mask]
        )
        ir_slopes[name] = float(
            np.polyfit(np.log(bundle.k_loop[low_mask]), np.log(ratio), 1)[0]
        )

    fastpt_summary = None
    fastpt_data = None
    if not args.skip_fastpt:
        try:
            fastpt_summary, fastpt_k, fastpt_residual = _compare_fastpt(
                bundle, iz, args.fastpt_n
            )
            fastpt_data = (fastpt_k, fastpt_residual)
        except ImportError as exc:
            fastpt_summary = {"unavailable": str(exc)}

    comparisons = []
    for path in args.compare_theory:
        other = load_theory(path)
        comparisons.append(
            {
                "file": str(path.resolve()),
                "digest": other.metadata["bundle_digest"],
                "metrics": _compare_bundles(bundle, other, args.redshift),
            }
        )

    pointwise_summary = {}
    for name in CHANNELS:
        rows = [row for row in point_rows if row["channel"] == name]
        pointwise_summary[name] = {
            "max_abs_p22_recomputed_minus_full_qp_reference_over_tree": float(
                max(
                    abs(row["p22_recomputed_minus_reference_over_tree"])
                    for row in rows
                )
            ),
            "max_abs_p13_analytic_minus_recursive_over_tree": float(
                max(
                    abs(row["p13_analytic_minus_recursive_over_tree"])
                    for row in rows
                )
            ),
        }

    summary = {
        "theory_file": str(args.theory.resolve()),
        "theory_digest": bundle.metadata["bundle_digest"],
        "redshift": float(bundle.z[iz]),
        "loop_weight": float(bundle.loop_weight),
        "algebra": {
            "max_abs_reconstruction_error": float(
                np.max(np.abs(reconstructed - bundle.channels_one_loop))
            ),
            "max_abs_p13_cross_identity_error": float(np.max(np.abs(cross_identity))),
            "minimum_p22_cauchy_margin": float(np.min(cauchy_margin)),
            "minimum_total_channel_k_le_ktrust": {
                name: float(
                    np.min(
                        bundle.channels_one_loop[
                            iz,
                            bundle.k_loop <= float(numerics["k_trust"]),
                            channel,
                        ]
                    )
                )
                for channel, name in enumerate(CHANNELS)
            },
        },
        "ir_log_slopes_abs_loop_over_tree_k_1e-3_to_1e-2": ir_slopes,
        "pointwise_independent_quadrature": pointwise_summary,
        "pointwise_settings": {
            "k_hmpc": sorted({float(row["k_hmpc"]) for row in point_rows}),
            "p22_full_qp_n_q_per_region": int(args.p22_reference_nq),
            "p22_full_qp_n_p": int(args.p22_reference_np),
            "p13_recursive_n_q": int(args.p13_recursive_nq),
            "p13_recursive_n_mu": int(args.p13_recursive_nmu),
            "p13_recursive_epsilon_relative": float(args.p13_recursive_epsilon),
        },
        "fastpt": fastpt_summary,
        "bundle_comparisons": comparisons,
        "pointwise_csv": str(csv_path.resolve()),
    }
    summary_path = args.output_dir / "raw_spt_validation_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)

    if not args.no_plots:
        _make_plots(args.output_dir, bundle, iz, fastpt_data)

    report_lines = [
        "# Raw one-loop SPT validation",
        "",
        f"Theory: `{args.theory}`",
        f"Redshift: `{bundle.z[iz]:g}`",
        "",
        "## Algebraic identities",
        "",
        f"- max bundle reconstruction error: `{summary['algebra']['max_abs_reconstruction_error']:.6e}`",
        f"- max P13 cross-channel identity error: `{summary['algebra']['max_abs_p13_cross_identity_error']:.6e}`",
        f"- minimum P22 Cauchy margin: `{summary['algebra']['minimum_p22_cauchy_margin']:.6e}`",
        "",
        "## Independent pointwise quadratures",
        "",
        "| channel | max abs delta P22 / tree | max abs delta P13 / tree |",
        "| --- | ---: | ---: |",
        *[
            f"| {name} | "
            f"{pointwise_summary[name]['max_abs_p22_recomputed_minus_full_qp_reference_over_tree']:.6e} | "
            f"{pointwise_summary[name]['max_abs_p13_analytic_minus_recursive_over_tree']:.6e} |"
            for name in CHANNELS
        ],
        "",
        "## Independent FAST-PT comparison",
        "",
    ]
    if fastpt_summary and "unavailable" not in fastpt_summary:
        report_lines.extend(
            [
                "| channel | max abs delta loop / tree | RMS delta loop / tree |",
                "| --- | ---: | ---: |",
                *[
                    f"| {name} | {fastpt_summary['max_abs_delta_loop_over_tree'][name]:.6e} | "
                    f"{fastpt_summary['rms_delta_loop_over_tree'][name]:.6e} |"
                    for name in CHANNELS
                ],
            ]
        )
    else:
        report_lines.append(f"FAST-PT unavailable: `{fastpt_summary}`")
    report_lines.extend(
        [
            "",
            "The P22/P13 split is not compared separately to FAST-PT because its IR-regularized decomposition differs; only their physical sum is comparable.",
            "",
            "Detailed pointwise values are stored in `raw_spt_pointwise.csv`.",
        ]
    )
    (args.output_dir / "raw_spt_validation_report.md").write_text(
        "\n".join(report_lines) + "\n"
    )
    print(f"saved: {summary_path.resolve()}")
    if fastpt_summary and "unavailable" not in fastpt_summary:
        for channel in CHANNELS:
            print(
                f"FAST-PT {channel:10s}: max |delta loop|/tree="
                f"{fastpt_summary['max_abs_delta_loop_over_tree'][channel]:.6e}, "
                f"RMS={fastpt_summary['rms_delta_loop_over_tree'][channel]:.6e}"
            )


if __name__ == "__main__":
    main()
