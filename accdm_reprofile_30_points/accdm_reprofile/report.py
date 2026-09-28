"""Small diagnostic reports from saved reprofiling tables; never run a fit.

All files produced here live in ``output_dir/reports``.  A missing result remains
missing in figures, including a planned point for which no fit was possible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PARAMETERS = (
    "log_alpha_F", "beta_F", "alpha_bias", "beta_bias", "alpha_ct", "beta_ct"
)
DEFAULT_REFERENCE = 188.429597466
MAIN_CUTS = (
    ("mass_f_m0p3", "log10f_acc", -0.3, "log10m_acc", 15.8571428571, 16.8571428571),
    ("mass_f_0", "log10f_acc", 0.0, "log10m_acc", 15.8571428571, 16.8571428571),
    ("fraction_m14p714", "log10m_acc", 14.7142857143, "log10f_acc", -0.4, 0.0),
    ("fraction_m16p429", "log10m_acc", 16.4285714286, "log10f_acc", -0.4, 0.0),
)
CENTRES = (
    (16.4285714286, -0.3), (16.4285714286, 0.0), (14.7142857143, -0.1)
)


def make_report(output_dir: Path, manifest_path: Path | None = None) -> list[Path]:
    """Write inexpensive PNG/CSV/JSON diagnostics and return the written paths.

    Input filenames are the standardized ``reprofile_*.csv`` files emitted by
    the campaign.  ``manifest_path`` may point to ``pilot_points.csv``.  No
    optional plotting dependency is imported until this function is called.
    Missing dependencies or input tables are recorded in ``report_status.json``.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    warnings: list[str] = []
    status: dict[str, Any] = {
        "input_directory": str(output_dir),
        "source": "saved result tables only; no fit or likelihood recomputation",
        "warnings": warnings,
    }

    def write_status() -> None:
        path = report_dir / "report_status.json"
        status["written_files"] = [p.name for p in written]
        path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        if path not in written:
            written.append(path)

    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        warnings.append(f"Numerical reporting dependency unavailable: {exc}")
        write_status()
        return written

    def read_table(path: Path, *, required: bool = False):
        if not path.is_file():
            if required:
                warnings.append(f"Missing input table: {path.name}")
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeError) as exc:
            warnings.append(f"Could not read {path.name}: {exc}")
            return pd.DataFrame()

    def normalize_points(frame):
        frame = frame.copy()
        if "target" not in frame and "coordinate_key" in frame:
            frame["target"] = frame["coordinate_key"]
        if "target" in frame:
            parts = frame["target"].astype(str).str.split("|", n=1, expand=True, regex=False)
            if len(parts.columns) == 2:
                for n, col in enumerate(("log10m_acc", "log10f_acc")):
                    if col not in frame:
                        frame[col] = pd.to_numeric(parts[n], errors="coerce")
        for col in ("log10m_acc", "log10f_acc", "chi2_old", "chi2_best", "gain"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if {"log10m_acc", "log10f_acc"}.issubset(frame.columns) and "target" not in frame:
            frame["target"] = [
                f"{m:.10f}|{f:.10f}" for m, f in zip(frame.log10m_acc, frame.log10f_acc)
            ]
        return frame

    points = normalize_points(read_table(output_dir / "reprofile_best_points.csv", required=True))
    attempts = read_table(output_dir / "reprofile_attempts.csv")
    if attempts.empty:
        # Some integrations keep this diagnostic table under the shorter name.
        attempts = read_table(output_dir / "attempts.csv")
    p1d = read_table(output_dir / "reprofile_p1d.csv")
    per_z = read_table(output_dir / "reprofile_chi2_by_redshift.csv")
    if manifest_path is None:
        candidates = [output_dir / "pilot_points.csv", output_dir.parent / "pilot_points.csv"]
        manifest_path = next((p for p in candidates if p.is_file()), None)
    manifest = normalize_points(read_table(Path(manifest_path))) if manifest_path is not None else pd.DataFrame()

    status.update({
        "n_saved_points": len(points), "n_manifest_points": len(manifest),
        "n_saved_attempts": len(attempts), "n_p1d_rows": len(p1d),
        "n_redshift_rows": len(per_z),
        "manifest": str(manifest_path) if manifest_path is not None else None,
    })

    def save_csv(frame, name: str) -> Path:
        path = report_dir / name
        frame.to_csv(path, index=False)
        written.append(path)
        return path

    if points.empty or "target" not in points:
        warnings.append("No saved best-point results. No fitted points or curves were inferred.")
        save_csv(pd.DataFrame([{"metric": "saved_points", "value": len(points)}]), "reprofile_summary.csv")
        write_status()
        return written

    # Duplicate targets are not silently collapsed by choosing the lowest chi2.
    # Report their existence and use the last saved row as the output snapshot.
    duplicate = points["target"].duplicated(keep=False)
    if duplicate.any():
        save_csv(points.loc[duplicate], "duplicate_result_targets.csv")
        warnings.append("Duplicate result targets; plots use the last saved row, not a minimum across rows.")
        points = points.drop_duplicates("target", keep="last").copy()
    for col in ("chi2_old", "chi2_best", "gain"):
        if col not in points:
            points[col] = np.nan
    recomputed_gain = points.chi2_old - points.chi2_best
    inconsistent_gain = points.gain.notna() & recomputed_gain.notna() & ~np.isclose(
        points.gain, recomputed_gain, atol=1e-6, rtol=1e-8
    )
    if inconsistent_gain.any():
        warnings.append(f"{int(inconsistent_gain.sum())} saved gains disagree with chi2_old-chi2_best; report uses the latter.")
    points["gain"] = recomputed_gain

    # A baseline-only result is not proof that an optimizer evaluated a new
    # vector.  Keep the selected origin and validation status visible in exports.
    invalid = pd.Series(False, index=points.index)
    if "validation_status" in points:
        invalid = points.validation_status.fillna("").astype(str).str.lower().ne("validated")
        status["validation_status_counts"] = points.validation_status.fillna("missing").astype(str).value_counts().to_dict()
    else:
        warnings.append("No validation_status column; finite saved values are shown, but validation is unknown.")
    if "optimization_status" in points:
        status["optimization_status_counts"] = points.optimization_status.fillna("missing").astype(str).value_counts().to_dict()
    points.loc[invalid, "chi2_best"] = np.nan
    points.loc[invalid, "gain"] = np.nan
    for col in points:
        if col.startswith("parameter_best_"):
            points.loc[invalid, col] = np.nan
    old_valid = np.isfinite(points.chi2_old)
    new_valid = np.isfinite(points.chi2_best)
    comparable = old_valid & new_valid
    improvement_tolerance = 1e-3 + 1e-6 * points.chi2_old.abs()
    improved = comparable & (points.gain > improvement_tolerance)

    refs = pd.to_numeric(points.get("chi2_reference", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(refs) and np.allclose(refs, refs.iloc[0], rtol=0, atol=1e-8):
        reference = float(refs.iloc[0])
        reference_source = "chi2_reference column"
    else:
        reference = DEFAULT_REFERENCE
        reference_source = "fixed original sampled reference"
        if len(refs):
            warnings.append("Inconsistent chi2_reference values; using the fixed original reference for all panels.")
    status.update({"chi2_reference": reference, "reference_source": reference_source})

    metrics: list[dict[str, Any]] = [
        {"metric": "manifest_points", "value": len(manifest)},
        {"metric": "saved_points", "value": len(points)},
        {"metric": "points_with_old_chi2", "value": int(old_valid.sum())},
        {"metric": "points_with_valid_best_chi2", "value": int(new_valid.sum())},
        {"metric": "comparable_points", "value": int(comparable.sum())},
        {"metric": "points_improved_beyond_numerical_tolerance", "value": int(improved.sum())},
        {"metric": "maximum_gain", "value": float(points.loc[comparable, "gain"].max()) if comparable.any() else np.nan},
        {"metric": "median_gain", "value": float(points.loc[comparable, "gain"].median()) if comparable.any() else np.nan},
        {"metric": "chi2_reference_for_all_plots", "value": reference},
        {"metric": "saved_attempts", "value": len(attempts)},
    ]
    if "selected_origin" in points:
        for name, count in points.selected_origin.fillna("missing").astype(str).value_counts().items():
            metrics.append({"metric": f"selected_origin:{name}", "value": int(count)})
    if "optimization_status" in points:
        for name, count in points.optimization_status.fillna("missing").astype(str).value_counts().items():
            metrics.append({"metric": f"optimization_status:{name}", "value": int(count)})
    save_csv(pd.DataFrame(metrics), "reprofile_summary.csv")
    summary_columns = [c for c in (
        "target", "log10m_acc", "log10f_acc", "chi2_old", "chi2_best", "gain",
        "selected_origin", "validation_status", "optimization_status"
    ) if c in points]
    save_csv(points[summary_columns].sort_values("gain", ascending=False, na_position="last"), "point_gain_summary.csv")
    if not manifest.empty and "target" in manifest:
        coverage = manifest[[c for c in ("target", "log10m_acc", "log10f_acc", "groups") if c in manifest]].copy()
        coverage["has_saved_result"] = coverage.target.isin(points.target)
        coverage["has_valid_best_chi2"] = coverage.target.isin(points.loc[new_valid, "target"])
        save_csv(coverage, "manifest_coverage.csv")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:
        warnings.append(f"Plotting unavailable; CSV summaries were written: {exc}")
        write_status()
        return written

    def save_figure(fig, name: str) -> None:
        path = report_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)

    def cut_rows(frame, cut):
        _, fixed, value, varying, lo, hi = cut
        if not {fixed, varying}.issubset(frame.columns):
            return pd.DataFrame()
        return frame[np.isclose(frame[fixed], value, rtol=0, atol=2e-7)
                     & frame[varying].between(lo - 2e-7, hi + 2e-7)].sort_values(varying)

    def cut_label(cut) -> str:
        _, fixed, value, varying, _, _ = cut
        symbol = r"f_{\rm acc}" if fixed == "log10f_acc" else r"m_{\rm acc}"
        return rf"$\log_{{10}}{symbol}={value:.4f}$"

    def axis_label(varying) -> str:
        return r"$\log_{10}m_{\rm acc}$" if varying == "log10m_acc" else r"$\log_{10}f_{\rm acc}$"

    def cut_values(cut, col):
        """Return observed values with NaNs at planned or regular-lattice gaps."""
        rows = cut_rows(points, cut)
        planned = cut_rows(manifest, cut)
        varying = cut[3]
        observed_x = rows[varying].to_numpy(float) if varying in rows else np.array([])
        planned_x = planned[varying].to_numpy(float) if varying in planned else np.array([])
        xs = np.unique(np.round(np.r_[observed_x, planned_x], 8))
        if not len(xs):
            return xs, np.array([])
        vals = pd.to_numeric(rows[col], errors="coerce").to_numpy(float) if col in rows else np.full(len(rows), np.nan)
        ys = []
        for x in xs:
            match = np.flatnonzero(np.isclose(observed_x, x, rtol=0, atol=2e-7))
            ys.append(vals[match[-1]] if len(match) else np.nan)
        # Explicit breaks prevent a line from visually filling the absent
        # logf=-0.3 point at logm=14.714... even if it is not in the manifest.
        expected_step = 1.0 / 7 if varying == "log10m_acc" else 0.1
        out_x, out_y = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i and x - xs[i-1] > 1.6 * expected_step:
                out_x.append((x + xs[i-1]) / 2)
                out_y.append(np.nan)
            out_x.append(x)
            out_y.append(y)
        return np.asarray(out_x), np.asarray(out_y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, cut in zip(axes.flat, MAIN_CUTS):
        for col, label, marker, color in (
            ("chi2_old", "zapisany fit", "o", "#6b7280"),
            ("chi2_best", "najlepszy zapisany wynik kampanii", "s", "#007c91"),
        ):
            x, y = cut_values(cut, col)
            ax.plot(x, y-reference, marker=marker, ms=4, lw=1.3, label=label, color=color)
        ax.axhline(0, color="black", lw=0.7, alpha=.4)
        ax.set(title=cut_label(cut), xlabel=axis_label(cut[3]), ylabel=r"$\chi^2-\chi^2_{\rm ref}$")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Profilowanie: ten sam punkt odniesienia χ²ref={reference:.9f}\nBrak wyniku = przerwa; linie łączą wyłącznie sąsiednie zapisane punkty", fontsize=12)
    save_figure(fig, "reprofile_chi2_cuts.png")

    if {"log10m_acc", "log10f_acc"}.issubset(points.columns):
        color_values = np.r_[points.loc[old_valid, "chi2_old"].to_numpy()-reference,
                             points.loc[new_valid, "chi2_best"].to_numpy()-reference]
        finite = color_values[np.isfinite(color_values)]
        if len(finite):
            vmin, vmax = min(0.0, float(finite.min())), float(finite.max())
            norm = Normalize(vmin=vmin, vmax=max(vmin+1e-8, vmax))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for ax, col, title in zip(axes, ("chi2_old", "chi2_best"), ("Przed kampanią", "Najlepsze zapisane wyniki")):
                if not manifest.empty and {"log10m_acc", "log10f_acc"}.issubset(manifest):
                    ax.scatter(manifest.log10m_acc, manifest.log10f_acc, marker="x", color=".7", s=25, label="planowane")
                valid = np.isfinite(points[col])
                im = ax.scatter(points.loc[valid, "log10m_acc"], points.loc[valid, "log10f_acc"],
                                c=points.loc[valid, col]-reference, cmap="viridis_r", norm=norm,
                                s=46, edgecolors="white", linewidths=.4)
                ax.set(title=title, xlabel=axis_label("log10m_acc"), ylabel=axis_label("log10f_acc"))
                ax.grid(alpha=.2)
            fig.colorbar(im, ax=axes, label=r"$\chi^2-\chi^2_{\rm ref}$ (wspólna skala)")
            fig.suptitle("Wyłącznie ocenione punkty; brak interpolacji powierzchni")
            save_figure(fig, "reprofile_before_after.png")

    # Compare canonical optimizer endpoints only. Retained / best-visited
    # candidates are useful for the best-point plot, not for convergence claims.
    # has_new_vector is deliberately irrelevant: a true stationary endpoint may
    # coincide with its starting vector.
    # A rerun with insufficient results must not leave an older plot suggesting
    # that convergence is supported by the current input snapshot.
    for name in ("direction_comparison.csv", "reprofile_direction_comparison.png"):
        stale = report_dir / name
        if stale.is_file():
            stale.unlink()
    direction_columns = {"target", "direction", "optimizer_success", "chi2_optimizer_final"}
    if not attempts.empty and direction_columns.issubset(attempts):
        directions = attempts.direction.fillna("").astype(str).str.lower().str.strip()
        direction_map = {"fwd": "forward", "forward": "forward", "back": "backward", "backward": "backward", "bwd": "backward"}
        work = attempts[directions.isin(direction_map)].copy()
        work["report_direction"] = directions[directions.isin(direction_map)].map(direction_map)
        success = work.optimizer_success.fillna(False).astype(str).str.lower().str.strip().isin(("true", "1", "1.0", "yes"))
        non_optimizer = pd.Series(False, index=work.index)
        for col in ("method", "stage"):
            if col in work:
                non_optimizer |= work[col].fillna("").astype(str).str.lower().str.strip().isin(
                    ("profiled_start", "candidate_old", "old", "baseline", "retention", "retained")
                )
        work["report_chi2"] = pd.to_numeric(work.chi2_optimizer_final, errors="coerce")
        work = work[success & ~non_optimizer & np.isfinite(work.report_chi2)].copy()
        status["n_converged_direction_attempts"] = len(work)
        status["direction_comparison_quantity"] = "minimum canonical chi2_optimizer_final among optimizer_success=True attempts per target and direction"
        if not work.empty:
            comparison = work.groupby(["target", "report_direction"], as_index=False).report_chi2.min().pivot(
                index="target", columns="report_direction", values="report_chi2"
            ).reset_index()
            if {"forward", "backward"}.issubset(comparison):
                comparison["backward_minus_forward"] = comparison.backward-comparison.forward
            save_csv(comparison, "direction_comparison.csv")
            common = comparison.dropna(subset=[c for c in ("forward", "backward") if c in comparison])
            if {"forward", "backward"}.issubset(common) and len(common):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
                axes[0].scatter(common.forward-reference, common.backward-reference, s=28, color="#007c91")
                lo = min(common.forward.min(), common.backward.min())-reference
                hi = max(common.forward.max(), common.backward.max())-reference
                axes[0].plot([lo, hi], [lo, hi], "--", color=".5", lw=1)
                axes[0].set(xlabel=r"forward: $\chi^2-\chi^2_{\rm ref}$", ylabel=r"backward: $\chi^2-\chi^2_{\rm ref}$")
                axes[1].bar(np.arange(len(common)), common.backward-common.forward, color="#007c91")
                axes[1].set(xlabel="indeks porównanego punktu", ylabel=r"$\chi^2_{backward}-\chi^2_{forward}$")
                for ax in axes:
                    ax.grid(alpha=.2)
                fig.suptitle("Końcowe wyniki prób zgłaszających zbieżność\nNajniższe końcowe χ² w każdym kierunku; zgodność nie dowodzi globalnego minimum", fontsize=11)
                save_figure(fig, "reprofile_direction_comparison.png")
            else:
                warnings.append("No targets with finite optimizer endpoints reporting success in both forward and backward directions; direction plot omitted.")
        else:
            warnings.append("No forward/backward optimizer endpoints reporting success; direction plot omitted.")
    else:
        warnings.append("Missing attempts or canonical endpoint/success columns; direction plot omitted.")

    for cut in MAIN_CUTS:
        if cut_rows(points, cut).empty:
            continue
        have_parameters = any(f"parameter_best_{p}" in points or f"parameter_old_{p}" in points for p in PARAMETERS)
        if not have_parameters:
            break
        fig = plt.figure(figsize=(14, 9.5), constrained_layout=True)
        panels = fig.add_gridspec(2, 3)
        nuisance_axes = []
        for slot, param in zip(panels, PARAMETERS):
            cell = slot.subgridspec(2, 1, height_ratios=(3.2, 1.1), hspace=.03)
            ax = fig.add_subplot(cell[0])
            position_ax = fig.add_subplot(cell[1], sharex=ax)
            nuisance_axes.append(ax)
            plt.setp(ax.get_xticklabels(), visible=False)
            for prefix, label, color, marker in (("parameter_old_", "stare", "#6b7280", "o"), ("parameter_best_", "najlepsze zapisane", "#007c91", "s")):
                x, y = cut_values(cut, prefix+param)
                ax.plot(x, y, marker=marker, color=color, ms=3.5, lw=1, label=label)
                required = {prefix+param, "lower_"+param, "upper_"+param}
                position_col = "_report_position_"+prefix+param
                if required.issubset(points):
                    lo = pd.to_numeric(points["lower_"+param], errors="coerce")
                    hi = pd.to_numeric(points["upper_"+param], errors="coerce")
                    width = (hi-lo).where(hi > lo)
                    points[position_col] = (pd.to_numeric(points[prefix+param], errors="coerce")-lo)/width
                    if prefix == "parameter_best_":
                        points.loc[invalid, position_col] = np.nan
                    xp, yp = cut_values(cut, position_col)
                    position_ax.plot(xp, yp, marker=marker, color=color, ms=3, lw=.8)
            # This scale uses each point's actual bounds, even when their widths
            # vary. Keeping it separate avoids flattening alpha_ct~0.1 against
            # numerical box bounds of +/-1000 on the parameter-value axis.
            position_ax.axhline(0, color="#ae3e29", lw=.7)
            position_ax.axhline(1, color="#ae3e29", lw=.7)
            position_ax.set(ylim=(-.06, 1.06), yticks=(0, .5, 1),
                            xlabel=axis_label(cut[3]), ylabel="pozycja")
            position_ax.tick_params(axis="y", labelsize=7)
            position_ax.grid(alpha=.15)
            ax.set(title=param)
            ax.grid(alpha=.2)
        nuisance_axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"Nuisance: {cut_label(cut)}\nWartości oraz pozycja (η−L)/(U−L) względem własnych granic każdego punktu: 0=dolna, 1=górna", fontsize=12)
        save_figure(fig, f"reprofile_nuisance_{cut[0]}.png")

    nuisance_columns = [c for c in points if c.startswith(("parameter_old_", "parameter_best_", "lower_", "upper_"))]
    if nuisance_columns:
        save_csv(points[["target", *nuisance_columns]], "nuisance_bound_summary.csv")

    # Select at most five informative P1D reports: three original centres and
    # two holes / large gains.  Selection never creates predictions for a gap.
    available = set(p1d.target.astype(str)) if "target" in p1d else set()
    chosen: list[str] = []
    for m, f in CENTRES:
        rows = points[np.isclose(points.get("log10m_acc", np.nan), m, atol=2e-7, rtol=0)
                      & np.isclose(points.get("log10f_acc", np.nan), f, atol=2e-7, rtol=0)]
        for target in rows.target.astype(str):
            if target in available and target not in chosen:
                chosen.append(target)
    ranked = points.sort_values("gain", ascending=False, na_position="last")
    hole_targets: set[str] = set()
    if not manifest.empty and "groups" in manifest and "target" in manifest:
        hole_targets = set(manifest.loc[manifest.groups.fillna("").astype(str).str.contains("hole", case=False), "target"].astype(str))
    additional = list(ranked.loc[ranked.target.isin(hole_targets), "target"].astype(str)) + list(ranked.target.astype(str))
    for target in additional:
        if len(chosen) >= 5:
            break
        if target in available and target not in chosen:
            chosen.append(target)
    status["p1d_report_targets"] = chosen

    needed_p1d = {"target", "z", "k_velocity", "data", "sigma", "prediction_old", "prediction_best"}
    if available and not needed_p1d.issubset(p1d):
        warnings.append(f"P1D plots skipped; missing columns: {sorted(needed_p1d-set(p1d.columns))}")
    elif needed_p1d.issubset(p1d):
        for target in chosen:
            rows = p1d[p1d.target.astype(str) == target].copy()
            for col in needed_p1d-{"target"}:
                rows[col] = pd.to_numeric(rows[col], errors="coerce")
            redshifts = sorted(rows.z.dropna().unique())
            if not redshifts:
                continue
            fig, axes = plt.subplots(2, len(redshifts), figsize=(max(10, 2.6*len(redshifts)), 6), squeeze=False, constrained_layout=True)
            for j, z in enumerate(redshifts):
                part = rows[np.isclose(rows.z, z)].sort_values("k_velocity")
                ax, res = axes[:, j]
                ax.errorbar(part.k_velocity, part.data, yerr=part.sigma, fmt=".", ms=3, lw=.5, color=".55", label="dane")
                for col, label, color in (("prediction_old", "stare", "#6b7280"), ("prediction_best", "najlepsze zapisane", "#007c91")):
                    ax.plot(part.k_velocity, part[col], color=color, lw=1.1, label=label)
                    safe_sigma = part.sigma.where(part.sigma > 0)
                    res.plot(part.k_velocity, (part[col]-part.data)/safe_sigma, color=color, lw=1.1)
                ax.set_title(f"z={z:g}")
                ax.set_xscale("log")
                res.set_xscale("log")
                res.axhline(0, color="black", lw=.5)
                res.set_xlabel(r"$k$ [s/km]")
                ax.grid(alpha=.2)
                res.grid(alpha=.2)
            axes[0, 0].set_ylabel("P1D [jednostki danych]")
            axes[1, 0].set_ylabel(r"$(P_{model}-P_{data})/\sigma$")
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"P1D i reszty: {target}\nWyłącznie zapisane predykcje; brak ponownego wywołania likelihood", fontsize=12)
            slug = target.replace("|", "_f").replace("-", "m").replace(".", "p")
            save_figure(fig, f"reprofile_p1d_m{slug}.png")

    required_z = {"target", "z", "chi2_old", "chi2_best"}
    if not per_z.empty and required_z.issubset(per_z):
        for col in ("z", "chi2_old", "chi2_best"):
            per_z[col] = pd.to_numeric(per_z[col], errors="coerce")
        per_z["gain"] = per_z.chi2_old-per_z.chi2_best
        save_csv(per_z, "redshift_gain_summary.csv")
        # Audit saved per-z sums against totals without recomputing the fit.
        summed = per_z.groupby("target")[["chi2_old", "chi2_best"]].sum(min_count=1).add_suffix("_sum_z").reset_index()
        audit = points[["target", "chi2_old", "chi2_best"]].merge(summed, on="target", how="inner")
        audit["old_sum_difference"] = audit.chi2_old_sum_z-audit.chi2_old
        audit["best_sum_difference"] = audit.chi2_best_sum_z-audit.chi2_best
        save_csv(audit, "redshift_sum_audit.csv")
        if (audit[["old_sum_difference", "best_sum_difference"]].abs() > 1e-5).any().any():
            warnings.append("Some saved per-redshift sums differ from totals; inspect redshift_sum_audit.csv (possible priors or missing bins).")
        selected_z = per_z[per_z.target.astype(str).isin(chosen)]
        if not selected_z.empty:
            fig, ax = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
            for target, group in selected_z.groupby("target", sort=False):
                group = group.sort_values("z")
                ax.plot(group.z, group.gain, "o-", ms=4, lw=1, label=target)
            ax.axhline(0, color="black", lw=.7)
            ax.set(xlabel="z", ylabel=r"$\chi^2_{old}(z)-\chi^2_{best}(z)$", title="Z których redshiftów pochodzi poprawa?")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            save_figure(fig, "reprofile_gain_by_redshift.png")

    write_status()
    return written
