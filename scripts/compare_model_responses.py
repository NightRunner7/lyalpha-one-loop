#!/usr/bin/env python3
"""Compare cosmological theory bundles before and after the P1D nuisance fit.

The diagnostic separates three effects:

1. the linear total-matter response relative to a reference theory;
2. the trusted one-loop channel response below ``k_trust``;
3. the observable P1D response with fixed and independently profiled nuisance
   parameters.

Inputs are saved theory bundles and fit JSON files, so this script never runs
CLASS or recomputes one-loop corrections.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from lyalpha_pt.data import DR12Dataset, load_dr12
from lyalpha_pt.fit import EffectiveModelFit, PARAMETER_NAMES
from lyalpha_pt.theory import CHANNELS, TheoryBundle, load_theory


@dataclass(frozen=True)
class ComparisonPoint:
    label: str
    theory_path: Path
    theory: TheoryBundle
    fit_path: Path | None
    fit: dict[str, Any] | None
    theta: np.ndarray | None


def parse_labeled_path(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=PATH`` without restricting characters in the label."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.")
    label, raw_path = value.rsplit("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Both LABEL and PATH must be non-empty.")
    return label, Path(raw_path)


def load_one_loop_fit(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        fit = payload["fits"]["one_loop"]
        parameters = fit["parameters"]
        theta = np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=float)
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} does not contain a complete one-loop fit.") from error
    if theta.shape != (6,) or not np.all(np.isfinite(theta)):
        raise ValueError(f"{path} contains an invalid nuisance-parameter vector.")
    return fit, theta


def infer_campaign_fit_path(theory_path: Path) -> Path | None:
    """Infer ``campaign/fits/POINT.json`` from ``campaign/theory/POINT.npz``."""

    if theory_path.parent.name != "theory":
        return None
    candidate = theory_path.parent.parent / "fits" / f"{theory_path.stem}.json"
    return candidate if candidate.is_file() else None


def fixed_nuisance_theta(
    reference_theta: Sequence[float],
    reference_fitter: EffectiveModelFit,
    target_fitter: EffectiveModelFit,
) -> np.ndarray:
    """Transfer nuisance parameters while preserving counterterm amplitude.

    ``alpha_ct`` is internally normalized by a theory-dependent ``i0_scale``.
    Rescaling it here keeps ``i0_scale * alpha_ct`` fixed, which makes the
    fixed-nuisance comparison physical rather than parameterization-dependent.
    """

    theta = np.asarray(reference_theta, dtype=float).copy()
    theta[4] *= reference_fitter.i0_scale / target_fitter.i0_scale
    return theta


def _redshift_index(bundle: TheoryBundle, redshift: float) -> int:
    matches = np.flatnonzero(np.isclose(bundle.z, redshift, rtol=0, atol=1e-9))
    if len(matches) != 1:
        raise ValueError(
            f"Theory bundle does not contain exactly one z={redshift:g} slice."
        )
    return int(matches[0])


def _interpolate(
    k_source: np.ndarray,
    values: np.ndarray,
    k_target: np.ndarray,
) -> np.ndarray:
    tolerance = 2e-12
    if (
        np.min(k_target) < float(k_source[0]) * (1 - tolerance)
        or np.max(k_target) > float(k_source[-1]) * (1 + tolerance)
    ):
        raise ValueError("Requested k range is outside a theory bundle.")
    spline = InterpolatedUnivariateSpline(np.log(k_source), values, k=3)
    return spline(np.log(np.clip(k_target, k_source[0], k_source[-1])))


def spectrum_at(
    bundle: TheoryBundle,
    redshift: float,
    k: np.ndarray,
    *,
    source: str,
    channel: int = 0,
) -> np.ndarray:
    iz = _redshift_index(bundle, redshift)
    if source == "linear":
        return _interpolate(bundle.k_input, bundle.p_total_input[iz], k)
    if source == "tree":
        return _interpolate(bundle.k_loop, bundle.p_tree[iz], k)
    if source == "one_loop":
        return _interpolate(bundle.k_loop, bundle.channels_one_loop[iz, :, channel], k)
    raise ValueError("source must be linear, tree, or one_loop")


def percent_response(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = float(np.max(np.abs(reference)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Reference values have zero or invalid scale.")
    valid = np.abs(reference) > 1e-13 * scale
    if not np.all(valid):
        raise ValueError("Reference values cross numerical zero in the comparison range.")
    return 100.0 * (target / reference - 1.0)


def response_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min_pct": float(np.min(values)),
        "max_pct": float(np.max(values)),
        "max_abs_pct": float(np.max(np.abs(values))),
        "rms_pct": float(np.sqrt(np.mean(values**2))),
    }


def _common_k(
    reference: TheoryBundle,
    points: Sequence[ComparisonPoint],
    *,
    source: str,
    k_max: float,
    size: int = 600,
) -> np.ndarray:
    bundles = [reference, *(point.theory for point in points)]
    grids = [bundle.k_input if source == "linear" else bundle.k_loop for bundle in bundles]
    lower = max(float(grid[0]) for grid in grids)
    upper = min(k_max, *(float(grid[-1]) for grid in grids))
    if not lower < upper:
        raise ValueError("Theory bundles have no common k range.")
    return np.geomspace(lower, upper, size)


def _direct_k_band(
    dataset: DR12Dataset,
    reference: TheoryBundle,
    redshift: float,
) -> tuple[float, float]:
    iz = _redshift_index(reference, redshift)
    mask = np.isclose(dataset.z, redshift, rtol=0, atol=1e-9)
    conversion = float(reference.velocity_to_hmpc[iz])
    k_parallel = conversion * dataset.k_velocity[mask]
    return float(np.min(k_parallel)), float(np.max(k_parallel))


def _shade_k_window(
    axis: plt.Axes,
    direct_band: tuple[float, float],
    k_plot_max: float,
    *,
    add_label: bool,
) -> None:
    k_low, k_high = direct_band
    axis.axvspan(
        k_low,
        min(k_high, k_plot_max),
        color="tab:green",
        alpha=0.09,
        label="Direct BOSS $k_\\parallel$ range" if add_label else None,
    )
    if k_high < k_plot_max:
        axis.axvspan(
            k_high,
            k_plot_max,
            color="0.5",
            alpha=0.07,
            label="P1D projection tail" if add_label else None,
        )


def _plot_linear_response(
    reference: TheoryBundle,
    points: Sequence[ComparisonPoint],
    dataset: DR12Dataset,
    redshifts: Sequence[float],
    k_max: float,
    output_path: Path,
) -> None:
    k = _common_k(reference, points, source="linear", k_max=k_max)
    figure, axes = plt.subplots(
        1, len(redshifts), figsize=(5.2 * len(redshifts), 4.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    for column, (axis, redshift) in enumerate(zip(axes, redshifts)):
        reference_values = spectrum_at(
            reference, redshift, k, source="linear"
        )
        for point in points:
            values = spectrum_at(point.theory, redshift, k, source="linear")
            axis.plot(k, percent_response(values, reference_values), label=point.label)
        _shade_k_window(
            axis,
            _direct_k_band(dataset, reference, redshift),
            float(k[-1]),
            add_label=column == 0,
        )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xscale("log")
        axis.set_xlim(k[0], k[-1])
        axis.set_title(f"z = {redshift:g}")
        axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"$100\,[P_{\rm lin}/P_{\rm lin}^{\Lambda\rm CDM}-1]\ [\%]$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    figure.suptitle("Linear total-matter response", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_one_loop_channels(
    reference: TheoryBundle,
    points: Sequence[ComparisonPoint],
    redshift: float,
    k_max: float,
    output_path: Path,
) -> None:
    k = _common_k(reference, points, source="one_loop", k_max=k_max)
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), sharex=True)
    for channel, (axis, name) in enumerate(zip(axes, CHANNELS)):
        reference_values = spectrum_at(
            reference, redshift, k, source="one_loop", channel=channel
        )
        for point in points:
            values = spectrum_at(
                point.theory, redshift, k, source="one_loop", channel=channel
            )
            axis.plot(k, percent_response(values, reference_values), label=point.label)
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xscale("log")
        axis.set_xlim(k[0], k[-1])
        axis.set_title(name)
        axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"$100\,[P_{XY}/P_{XY}^{\Lambda\rm CDM}-1]\ [\%]$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    figure.suptitle(
        f"Trusted one-loop channel response at z = {redshift:g}", y=1.02
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_p1d_response(
    dataset: DR12Dataset,
    reference_model: np.ndarray,
    point_models: dict[str, tuple[np.ndarray, np.ndarray | None]],
    redshifts: Sequence[float],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        len(redshifts),
        figsize=(5.2 * len(redshifts), 8.0),
        sharex="col",
        sharey="row",
    )
    axes = np.asarray(axes).reshape(2, len(redshifts))
    for column, redshift in enumerate(redshifts):
        mask = np.isclose(dataset.z, redshift, rtol=0, atol=1e-9)
        k_velocity = dataset.k_velocity[mask]
        reference_slice = reference_model[mask]
        for label, (fixed_model, profiled_model) in point_models.items():
            axes[0, column].plot(
                k_velocity,
                percent_response(fixed_model[mask], reference_slice),
                label=label,
            )
            if profiled_model is not None:
                axes[1, column].plot(
                    k_velocity,
                    percent_response(profiled_model[mask], reference_slice),
                    label=label,
                )
        for row in range(2):
            axes[row, column].axhline(0, color="black", lw=0.8)
            axes[row, column].grid(alpha=0.18)
        axes[0, column].set_title(f"z = {redshift:g}")
        axes[1, column].set_xlabel(r"$k_v\ [\mathrm{s\,km}^{-1}]$")
    axes[0, 0].set_ylabel("Fixed nuisance response [%]")
    axes[1, 0].set_ylabel("Independently profiled response [%]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    figure.suptitle("Observable P1D response relative to the LCDM best-fit curve", y=1.01)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _load_points(
    point_specs: Sequence[tuple[str, Path]],
    explicit_fit_paths: dict[str, Path],
) -> list[ComparisonPoint]:
    points: list[ComparisonPoint] = []
    for label, theory_path in point_specs:
        if not theory_path.is_file():
            raise FileNotFoundError(theory_path)
        fit_path = explicit_fit_paths.get(label) or infer_campaign_fit_path(theory_path)
        fit: dict[str, Any] | None = None
        theta: np.ndarray | None = None
        if fit_path is not None:
            if not fit_path.is_file():
                raise FileNotFoundError(fit_path)
            fit, theta = load_one_loop_fit(fit_path)
        points.append(
            ComparisonPoint(
                label=label,
                theory_path=theory_path.resolve(),
                theory=load_theory(theory_path),
                fit_path=fit_path.resolve() if fit_path is not None else None,
                fit=fit,
                theta=theta,
            )
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-theory", type=Path, required=True)
    parser.add_argument("--reference-fit", type=Path)
    parser.add_argument(
        "--point",
        action="append",
        type=parse_labeled_path,
        required=True,
        metavar="LABEL=THEORY_NPZ",
    )
    parser.add_argument(
        "--point-fit",
        action="append",
        type=parse_labeled_path,
        default=[],
        metavar="LABEL=FIT_JSON",
        help="Override automatic campaign fit-path inference.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--redshifts", nargs="+", type=float, default=[3.0, 3.6, 4.2])
    parser.add_argument("--pivot-z", type=float, default=3.6)
    parser.add_argument("--linear-k-max", type=float, default=20.0)
    parser.add_argument("--trusted-k-max", type=float, default=2.0)
    parser.add_argument("--k-uv-cut", type=float, default=20.0)
    parser.add_argument("--covariance", default="paper_diag")
    args = parser.parse_args()

    labels = [label for label, _ in args.point]
    if len(labels) != len(set(labels)):
        raise ValueError("Every --point label must be unique.")
    fit_overrides = dict(args.point_fit)
    unknown_fit_labels = sorted(set(fit_overrides) - set(labels))
    if unknown_fit_labels:
        raise ValueError(f"--point-fit has unknown labels: {unknown_fit_labels}")

    reference = load_theory(args.reference_theory)
    points = _load_points(args.point, fit_overrides)
    dataset = load_dr12(args.data_dir)
    if not np.allclose(reference.z, dataset.z_unique, rtol=0, atol=1e-9):
        raise ValueError("Reference-theory and data redshift grids differ.")
    for point in points:
        if not np.allclose(point.theory.z, reference.z, rtol=0, atol=1e-9):
            raise ValueError(f"{point.label} and the reference use different redshifts.")

    redshifts = [float(value) for value in args.redshifts]
    for redshift in [*redshifts, float(args.pivot_z)]:
        _redshift_index(reference, redshift)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_figure = args.output_dir / "linear_matter_response.png"
    loop_figure = args.output_dir / "one_loop_channel_response.png"
    _plot_linear_response(
        reference,
        points,
        dataset,
        redshifts,
        args.linear_k_max,
        linear_figure,
    )
    _plot_one_loop_channels(
        reference,
        points,
        args.pivot_z,
        args.trusted_k_max,
        loop_figure,
    )

    reference_fit: dict[str, Any] | None = None
    reference_theta: np.ndarray | None = None
    reference_fitter: EffectiveModelFit | None = None
    reference_model: np.ndarray | None = None
    point_models: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    p1d_figure: Path | None = None
    if args.reference_fit is not None:
        reference_fit, reference_theta = load_one_loop_fit(args.reference_fit)
        reference_fitter = EffectiveModelFit(
            dataset,
            reference,
            mode="one_loop",
            k_uv_cut=args.k_uv_cut,
            covariance_mode=args.covariance,
        )
        reference_model = reference_fitter.model(reference_theta)
        for point in points:
            fitter = EffectiveModelFit(
                dataset,
                point.theory,
                mode="one_loop",
                k_uv_cut=args.k_uv_cut,
                covariance_mode=args.covariance,
            )
            transferred_theta = fixed_nuisance_theta(
                reference_theta, reference_fitter, fitter
            )
            fixed_model = fitter.model(transferred_theta)
            profiled_model = fitter.model(point.theta) if point.theta is not None else None
            point_models[point.label] = (fixed_model, profiled_model)
        p1d_figure = args.output_dir / "p1d_response_fixed_and_profiled.png"
        _plot_p1d_response(
            dataset,
            reference_model,
            point_models,
            redshifts,
            p1d_figure,
        )

    summary_rows: list[dict[str, Any]] = []
    k_linear = _common_k(
        reference, points, source="linear", k_max=args.linear_k_max, size=900
    )
    k_loop = _common_k(
        reference, points, source="one_loop", k_max=args.trusted_k_max, size=500
    )
    reference_chi2 = (
        float(reference_fit["chi2"]) if reference_fit is not None else np.nan
    )
    for point in points:
        for redshift in redshifts:
            direct_low, direct_high = _direct_k_band(dataset, reference, redshift)
            direct_linear = (k_linear >= direct_low) & (k_linear <= direct_high)
            direct_loop = (k_loop >= direct_low) & (k_loop <= direct_high)
            if not np.any(direct_linear) or not np.any(direct_loop):
                raise ValueError(f"No sampled direct-k values at z={redshift:g}.")

            reference_linear = spectrum_at(
                reference, redshift, k_linear, source="linear"
            )
            point_linear = spectrum_at(
                point.theory, redshift, k_linear, source="linear"
            )
            linear_response = percent_response(point_linear, reference_linear)

            reference_loop = spectrum_at(
                reference, redshift, k_loop, source="one_loop", channel=0
            )
            point_loop = spectrum_at(
                point.theory, redshift, k_loop, source="one_loop", channel=0
            )
            loop_response = percent_response(point_loop, reference_loop)

            linear_stats = response_statistics(linear_response[direct_linear])
            loop_stats = response_statistics(loop_response[direct_loop])
            row: dict[str, Any] = {
                "label": point.label,
                "redshift": redshift,
                "theory_path": str(point.theory_path),
                "fit_path": str(point.fit_path) if point.fit_path is not None else "",
                "direct_k_min_hmpc": direct_low,
                "direct_k_max_hmpc": direct_high,
                **{f"linear_{key}": value for key, value in linear_stats.items()},
                **{f"one_loop_dd_{key}": value for key, value in loop_stats.items()},
                "chi2": float(point.fit["chi2"]) if point.fit is not None else np.nan,
                "delta_chi2_vs_reference": (
                    float(point.fit["chi2"]) - reference_chi2
                    if point.fit is not None and np.isfinite(reference_chi2)
                    else np.nan
                ),
            }
            if reference_model is not None:
                mask = np.isclose(dataset.z, redshift, rtol=0, atol=1e-9)
                fixed_model, profiled_model = point_models[point.label]
                fixed_response = percent_response(
                    fixed_model[mask], reference_model[mask]
                )
                fixed_stats = response_statistics(fixed_response)
                row.update(
                    {f"p1d_fixed_{key}": value for key, value in fixed_stats.items()}
                )
                if profiled_model is not None:
                    profiled_response = percent_response(
                        profiled_model[mask], reference_model[mask]
                    )
                    profiled_stats = response_statistics(profiled_response)
                    row.update(
                        {
                            f"p1d_profiled_{key}": value
                            for key, value in profiled_stats.items()
                        }
                    )
            summary_rows.append(row)

    summary_path = args.output_dir / "model_response_summary.csv"
    fieldnames = list(summary_rows[0])
    for row in summary_rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata_path = args.output_dir / "model_response_metadata.json"
    metadata = {
        "reference_theory": str(args.reference_theory.resolve()),
        "reference_fit": (
            str(args.reference_fit.resolve()) if args.reference_fit is not None else None
        ),
        "points": [
            {
                "label": point.label,
                "theory_path": str(point.theory_path),
                "fit_path": str(point.fit_path) if point.fit_path is not None else None,
            }
            for point in points
        ],
        "redshifts": redshifts,
        "pivot_z": args.pivot_z,
        "linear_k_max_hmpc": args.linear_k_max,
        "trusted_k_max_hmpc": args.trusted_k_max,
        "k_uv_cut_hmpc": args.k_uv_cut,
        "covariance": args.covariance,
        "fixed_nuisance_counterterm_convention": (
            "alpha_ct is rescaled to preserve i0_scale * alpha_ct"
        ),
        "outputs": {
            "linear_figure": str(linear_figure.resolve()),
            "one_loop_figure": str(loop_figure.resolve()),
            "p1d_figure": str(p1d_figure.resolve()) if p1d_figure else None,
            "summary_csv": str(summary_path.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"saved: {linear_figure.resolve()}")
    print(f"saved: {loop_figure.resolve()}")
    if p1d_figure is not None:
        print(f"saved: {p1d_figure.resolve()}")
    print(f"saved: {summary_path.resolve()}")
    print(f"saved: {metadata_path.resolve()}")
    print("\nMaximum absolute responses in the direct BOSS k_parallel window:")
    for row in summary_rows:
        line = (
            f"{row['label']:24s} z={row['redshift']:.1f} "
            f"linear={row['linear_max_abs_pct']:.4g}% "
            f"one-loop-dd={row['one_loop_dd_max_abs_pct']:.4g}%"
        )
        if "p1d_fixed_max_abs_pct" in row:
            line += f" P1D-fixed={row['p1d_fixed_max_abs_pct']:.4g}%"
        if "p1d_profiled_max_abs_pct" in row:
            line += f" P1D-profiled={row['p1d_profiled_max_abs_pct']:.4g}%"
        print(line)


if __name__ == "__main__":
    main()
