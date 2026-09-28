#!/usr/bin/env python3
"""Targeted convergence audit of the one-loop density--theta channel.

The script reuses the linear spectrum stored in a theory bundle.  If a q-max
scan needs more linear-spectrum support than the bundle contains, ``auto``
mode recomputes only the linear CLASS spectrum from the recorded CLASS input;
it never regenerates a full theory bundle or performs a P1D fit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from lyalpha_pt.spt import (
    p13_channels_richardson,
    p13_dtheta_analytic,
    p22_channels,
)
from lyalpha_pt.theory import load_theory


DEFAULT_K = (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0)
DEFAULT_QMAX = (50.0, 75.0, 100.0, 150.0)


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _spectrum_callable(k_grid: np.ndarray, values: np.ndarray):
    spline = InterpolatedUnivariateSpline(
        np.log(np.asarray(k_grid)), np.log(np.asarray(values)), k=3
    )

    def evaluate(k):
        return np.exp(spline(np.log(np.asarray(k, dtype=float))))

    return evaluate


def _class_extended_spectrum(bundle, redshift: float, required_k_max: float):
    try:
        from classy import Class
    except ImportError as exc:
        raise RuntimeError(
            "The q-max scan needs P_linear beyond the bundle range. Install or "
            "activate the same CLASS wrapper, or restrict --q-maxes so that "
            "q_max + max(k) does not exceed the bundle input range."
        ) from exc

    params = dict(bundle.metadata["class_params"])
    params["P_k_max_h/Mpc"] = 1.05 * required_k_max
    params["z_max_pk"] = max(float(params.get("z_max_pk", 0.0)), redshift + 0.2)
    params["output"] = "mPk"
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    h = float(cosmo.h())
    if not np.isclose(h, bundle.h, rtol=1e-10, atol=0):
        raise ValueError(f"CLASS h={h} differs from bundle h={bundle.h}.")

    source = str(bundle.metadata["loop_source"])
    if source == "total":
        method = cosmo.pk_lin
    elif source == "cb" and hasattr(cosmo, "pk_cb_lin"):
        method = cosmo.pk_cb_lin
    else:
        raise RuntimeError(
            f"The active CLASS wrapper cannot provide loop source {source!r}."
        )

    n_input = max(int(bundle.metadata["numerics"]["n_linear_input"]), 7000)
    k_grid = np.geomspace(float(bundle.k_input[0]), required_k_max, n_input)
    values = np.array([method(float(k * h), redshift) for k in k_grid]) * h**3

    if hasattr(cosmo, "struct_cleanup"):
        cosmo.struct_cleanup()
    if hasattr(cosmo, "empty"):
        cosmo.empty()

    iz = int(np.argmin(np.abs(bundle.z - redshift)))
    stored = bundle.p_loop_input[iz]
    overlap_spline = _spectrum_callable(k_grid, values)
    overlap = overlap_spline(bundle.k_input)
    overlap_relative = (overlap - stored) / stored
    return k_grid, values, {
        "kind": "class_extended",
        "required_k_max_hmpc": required_k_max,
        "max_abs_overlap_relative": float(np.max(np.abs(overlap_relative))),
        "rms_overlap_relative": float(np.sqrt(np.mean(overlap_relative**2))),
    }


def _linear_input(bundle, redshift: float, required_k_max: float, mode: str):
    iz = int(np.argmin(np.abs(bundle.z - redshift)))
    if not np.isclose(bundle.z[iz], redshift, rtol=0, atol=1e-9):
        raise ValueError(
            f"Requested z={redshift} is absent; available values are {bundle.z.tolist()}."
        )
    bundle_max = float(bundle.k_input[-1])
    if mode == "bundle" or (mode == "auto" and required_k_max <= bundle_max):
        if required_k_max > bundle_max:
            raise ValueError(
                f"Need linear P(k) to {required_k_max:g} h/Mpc, but the bundle "
                f"ends at {bundle_max:g} h/Mpc. Use --linear-source auto or class."
            )
        return (
            bundle.k_input.copy(),
            bundle.p_loop_input[iz].copy(),
            {"kind": "bundle", "required_k_max_hmpc": required_k_max},
        )
    return _class_extended_spectrum(bundle, redshift, required_k_max)


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    p_linear = _spectrum_callable(task["linear_k"], task["linear_p"])
    kind = task["kind"]
    result = {key: value for key, value in task.items() if key not in ("linear_k", "linear_p")}
    k = float(task["k_hmpc"])
    if kind in ("joint", "p22"):
        p22 = p22_channels(
            k,
            p_linear,
            q_min=float(task["q_min"]),
            q_max=float(task["q_max"]),
            n_q_low=int(task["n_q_low"]),
            n_q_mid=int(task["n_q_mid"]),
            n_q_high=int(task["n_q_high"]),
            n_p=int(task["n_p"]),
            middle_width=float(task["middle_width"]),
        )
        result["p22_dtheta"] = float(p22[1])
    if kind in ("joint", "p13", "analytic"):
        p13, error = p13_channels_richardson(
            k,
            p_linear,
            q_min=float(task["q_min"]),
            q_max=float(task["q_max"]),
            n_q=int(task["p13_n_q"]),
            n_mu=int(task["p13_n_mu"]),
            epsilon_relative=float(task["epsilon_relative"]),
        )
        result["p13_dtheta"] = float(p13[1])
        result["p13_richardson_error"] = float(error[1])
    if kind in ("joint", "analytic"):
        result["p13_analytic_dtheta"] = p13_dtheta_analytic(
            k,
            p_linear,
            q_min=float(task["q_min"]),
            q_max=float(task["q_max"]),
            n_q=int(task["analytic_n_q"]),
        )
    return result


def _task_base(linear_k, linear_p, k, q_min, q_max, numerics):
    return {
        "linear_k": linear_k,
        "linear_p": linear_p,
        "k_hmpc": float(k),
        "q_min": float(q_min),
        "q_max": float(q_max),
        "middle_width": float(numerics["middle_width"]),
    }


def _build_tasks(linear_k, linear_p, k_values, q_maxes, numerics):
    q_min = float(numerics["q_min"])
    p22_prod = {
        key: int(numerics[key])
        for key in ("n_q_low", "n_q_mid", "n_q_high", "n_p")
    }
    p13_prod = {
        "p13_n_q": int(numerics["p13_n_q"]),
        "p13_n_mu": int(numerics["p13_n_mu"]),
        "epsilon_relative": float(numerics["p13_epsilon_relative"]),
    }
    tasks = []

    for q_max in q_maxes:
        for k in k_values:
            task = _task_base(linear_k, linear_p, k, q_min, q_max, numerics)
            task.update(
                kind="joint",
                family="qmax",
                config=f"qmax_{q_max:g}",
                analytic_n_q=12001,
                **p22_prod,
                **p13_prod,
            )
            tasks.append(task)

    p22_variants = {
        "p22_coarse": {
            "n_q_low": max(80, p22_prod["n_q_low"] // 2),
            "n_q_mid": max(160, p22_prod["n_q_mid"] // 2),
            "n_q_high": max(80, p22_prod["n_q_high"] // 2),
            "n_p": max(48, 2 * p22_prod["n_p"] // 3),
        },
        "p22_fine": {
            "n_q_low": 3 * p22_prod["n_q_low"] // 2,
            "n_q_mid": 3 * p22_prod["n_q_mid"] // 2,
            "n_q_high": 3 * p22_prod["n_q_high"] // 2,
            "n_p": 4 * p22_prod["n_p"] // 3,
        },
    }
    for label, settings in p22_variants.items():
        for k in k_values:
            task = _task_base(linear_k, linear_p, k, q_min, q_maxes[0], numerics)
            task.update(kind="p22", family="p22_numerics", config=label, **settings)
            tasks.append(task)

    p13_variants = {
        "p13_coarse": {
            "p13_n_q": max(60, 2 * p13_prod["p13_n_q"] // 3),
            "p13_n_mu": max(12, 2 * p13_prod["p13_n_mu"] // 3),
            "epsilon_relative": p13_prod["epsilon_relative"],
        },
        "p13_fine": {
            "p13_n_q": 3 * p13_prod["p13_n_q"] // 2,
            "p13_n_mu": max(32, 5 * p13_prod["p13_n_mu"] // 3),
            "epsilon_relative": p13_prod["epsilon_relative"],
        },
        "epsilon_3e-4": {**p13_prod, "epsilon_relative": 3e-4},
        "epsilon_3e-5": {**p13_prod, "epsilon_relative": 3e-5},
    }
    for label, settings in p13_variants.items():
        for k in k_values:
            task = _task_base(linear_k, linear_p, k, q_min, q_maxes[0], numerics)
            task.update(kind="p13", family="p13_numerics", config=label, **settings)
            tasks.append(task)

    analytic_settings = p13_variants["p13_fine"]
    for k in k_values:
        task = _task_base(
            linear_k, linear_p, k, q_min, q_maxes[0], numerics
        )
        task.update(
            kind="analytic",
            family="analytic_p13",
            config="direct_vs_closed",
            analytic_n_q=12001,
            **analytic_settings,
        )
        tasks.append(task)
    return tasks, p22_prod, p13_prod


def _run_tasks(tasks, jobs: int):
    results = []
    total = len(tasks)
    if jobs == 1:
        for index, task in enumerate(tasks, 1):
            results.append(_worker(task))
            if index == total or index % 5 == 0:
                print(f"completed {index}/{total} numerical tasks", flush=True)
        return results

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index == total or index % 5 == 0:
                print(f"completed {index}/{total} numerical tasks", flush=True)
    return results


def _by_key(rows):
    return {
        (row["family"], row["config"], float(row["k_hmpc"])): row for row in rows
    }


def _complete_rows(rows, k_values, q_maxes):
    lookup = _by_key(rows)
    reference_qmax = f"qmax_{q_maxes[0]:g}"
    completed = []
    for row in rows:
        item = dict(row)
        k = float(item["k_hmpc"])
        base = lookup[("qmax", reference_qmax, k)]
        if item["family"] == "p22_numerics":
            item["p13_dtheta"] = base["p13_dtheta"]
            item["p13_richardson_error"] = base["p13_richardson_error"]
        elif item["family"] == "p13_numerics":
            item["p22_dtheta"] = base["p22_dtheta"]
        if "p22_dtheta" in item and "p13_dtheta" in item:
            item["loop_sum_dtheta"] = item["p22_dtheta"] + item["p13_dtheta"]
            denominator = max(abs(item["loop_sum_dtheta"]), 1e-300)
            item["cancellation_factor"] = (
                abs(item["p22_dtheta"]) + abs(item["p13_dtheta"])
            ) / denominator
        if "p22_dtheta" in item and "p13_analytic_dtheta" in item:
            item["loop_sum_closed_p13"] = (
                item["p22_dtheta"] + item["p13_analytic_dtheta"]
            )
        if "p13_analytic_dtheta" in item:
            scale = max(abs(item["p13_analytic_dtheta"]), 1e-300)
            item["p13_direct_minus_analytic_relative"] = (
                item["p13_dtheta"] - item["p13_analytic_dtheta"]
            ) / scale
        completed.append(item)
    # Reuse the qmax-first calculation as the explicit production P22 row so
    # its accuracy can be compared directly with the fine P22 quadrature.
    for k in k_values:
        item = dict(lookup[("qmax", reference_qmax, float(k))])
        item["family"] = "p22_numerics"
        item["config"] = "p22_production"
        item["loop_sum_dtheta"] = item["p22_dtheta"] + item["p13_dtheta"]
        denominator = max(abs(item["loop_sum_dtheta"]), 1e-300)
        item["cancellation_factor"] = (
            abs(item["p22_dtheta"]) + abs(item["p13_dtheta"])
        ) / denominator
        completed.append(item)
    return sorted(completed, key=lambda row: (row["family"], row["config"], row["k_hmpc"]))


def _relative_stats(values, reference, mask):
    values = np.asarray(values)[mask]
    reference = np.asarray(reference)[mask]
    scale = np.maximum(np.abs(reference), 1e-300)
    relative = (values - reference) / scale
    return {
        "rms_relative": float(np.sqrt(np.mean(relative**2))),
        "max_abs_relative": float(np.max(np.abs(relative))),
    }


def _summaries(rows, k_values, q_maxes):
    lookup = _by_key(rows)
    k = np.asarray(k_values, dtype=float)
    masks = {"k_le_2": k <= 2.0, "all_k": np.ones_like(k, dtype=bool)}
    specifications = [
        ("qmax", [f"qmax_{value:g}" for value in q_maxes], f"qmax_{q_maxes[-1]:g}"),
        (
            "p22_numerics",
            ["p22_coarse", "p22_production", "p22_fine"],
            "p22_fine",
        ),
        (
            "p13_numerics",
            ["p13_coarse", "p13_fine", "epsilon_3e-4", "epsilon_3e-5"],
            "p13_fine",
        ),
    ]
    summary = []
    for family, configs, reference_config in specifications:
        reference_rows = [lookup[(family, reference_config, float(value))] for value in k]
        components = ["loop_sum_dtheta"] if family == "qmax" else []
        components += ["p22_dtheta"] if family == "p22_numerics" else []
        components += ["p13_dtheta", "loop_sum_dtheta"] if family == "p13_numerics" else []
        if family == "qmax":
            components = [
                "p22_dtheta",
                "p13_dtheta",
                "loop_sum_dtheta",
                "p13_analytic_dtheta",
                "loop_sum_closed_p13",
            ]
        for config in configs:
            config_rows = [lookup[(family, config, float(value))] for value in k]
            for component in components:
                values = [row[component] for row in config_rows]
                reference = [row[component] for row in reference_rows]
                for range_name, mask in masks.items():
                    summary.append(
                        {
                            "family": family,
                            "config": config,
                            "reference": reference_config,
                            "component": component,
                            "range": range_name,
                            **_relative_stats(values, reference, mask),
                        }
                    )

    analytic = [
        lookup[("analytic_p13", "direct_vs_closed", float(value))] for value in k
    ]
    relative = np.array(
        [row["p13_direct_minus_analytic_relative"] for row in analytic]
    )
    for range_name, mask in masks.items():
        summary.append(
            {
                "family": "analytic_p13",
                "config": "direct_richardson",
                "reference": "closed_1d",
                "component": "p13_dtheta",
                "range": range_name,
                "rms_relative": float(np.sqrt(np.mean(relative[mask] ** 2))),
                "max_abs_relative": float(np.max(np.abs(relative[mask]))),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key not in ("kind",):
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, summaries, k_values, q_maxes, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = _by_key(rows)
    k = np.asarray(k_values, dtype=float)
    base = [lookup[("qmax", f"qmax_{q_maxes[0]:g}", float(value))] for value in k]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].loglog(k, np.abs([row["p22_dtheta"] for row in base]), "o-", label=r"$|P_{22}^{\delta\theta}|$")
    axes[0].loglog(k, np.abs([row["p13_dtheta"] for row in base]), "s-", label=r"$|P_{13}^{\delta\theta}|$")
    axes[0].loglog(k, np.abs([row["loop_sum_dtheta"] for row in base]), "^-", label=r"$|P_{22}+P_{13}|$")
    axes[0].axvline(2.0, color="0.5", ls="--", lw=1)
    axes[0].set(xlabel=r"$k\ [h/{\rm Mpc}]$", ylabel=r"absolute spectrum", title=rf"$z=3$, $q_{{\max}}={q_maxes[0]:g}$")
    axes[0].legend()
    axes[0].grid(alpha=0.25, which="both")
    axes[1].loglog(k, [row["cancellation_factor"] for row in base], "o-", color="tab:red")
    axes[1].axvline(2.0, color="0.5", ls="--", lw=1)
    axes[1].set(xlabel=r"$k\ [h/{\rm Mpc}]$", ylabel=r"$(|P_{22}|+|P_{13}|)/|P_{22}+P_{13}|$", title="cancellation amplification")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "dtheta_components_and_cancellation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ref_config = f"qmax_{q_maxes[-1]:g}"
    reference = [lookup[("qmax", ref_config, float(value))]["loop_sum_closed_p13"] for value in k]
    for q_max in q_maxes[:-1]:
        values = [lookup[("qmax", f"qmax_{q_max:g}", float(value))]["loop_sum_closed_p13"] for value in k]
        relative = (np.asarray(values) - reference) / np.maximum(np.abs(reference), 1e-300)
        axes[0].semilogx(k, relative, "o-", label=rf"$q_{{\max}}={q_max:g}$")
    axes[0].axvline(2.0, color="0.5", ls="--", lw=1)
    axes[0].axhline(0.0, color="0.5", lw=1)
    axes[0].set(xlabel=r"$k\ [h/{\rm Mpc}]$", ylabel=rf"relative to $q_{{\max}}={q_maxes[-1]:g}$", title=r"$P_{22}^{\delta\theta}+P_{13}^{\delta\theta}$ (closed $P_{13}$)")
    axes[0].legend()
    axes[0].grid(alpha=0.25, which="both")

    analytic = [lookup[("analytic_p13", "direct_vs_closed", float(value))] for value in k]
    axes[1].semilogx(k, [row["p13_direct_minus_analytic_relative"] for row in analytic], "o-", color="tab:purple")
    axes[1].axvline(2.0, color="0.5", ls="--", lw=1)
    axes[1].axhline(0.0, color="0.5", lw=1)
    axes[1].set(xlabel=r"$k\ [h/{\rm Mpc}]$", ylabel="(direct - closed) / |closed|", title=r"independent $P_{13}^{\delta\theta}$ check")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "dtheta_qmax_and_analytic_check.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    summary_lookup = {
        (row["family"], row["config"], row["component"], row["range"]): row
        for row in summaries
    }
    selections = [
        ("qmax", "qmax_50", "p22_dtheta", "qmax 50: P22"),
        ("qmax", "qmax_50", "p13_dtheta", "qmax 50: P13 direct"),
        ("qmax", "qmax_50", "loop_sum_dtheta", "qmax 50: sum direct"),
        ("qmax", "qmax_50", "loop_sum_closed_p13", "qmax 50: sum closed"),
        ("qmax", "qmax_75", "loop_sum_closed_p13", "qmax 75: sum closed"),
        ("qmax", "qmax_100", "loop_sum_closed_p13", "qmax 100: sum closed"),
        ("p22_numerics", "p22_production", "p22_dtheta", "P22 production vs fine"),
        ("p22_numerics", "p22_coarse", "p22_dtheta", "P22 coarse vs fine"),
        ("p13_numerics", "p13_coarse", "p13_dtheta", "P13 coarse vs fine"),
        ("p13_numerics", "p13_coarse", "loop_sum_dtheta", "sum: P13 coarse"),
        ("p13_numerics", "epsilon_3e-4", "loop_sum_dtheta", "sum: epsilon 3e-4"),
        ("p13_numerics", "epsilon_3e-5", "loop_sum_dtheta", "sum: epsilon 3e-5"),
        ("analytic_p13", "direct_richardson", "p13_dtheta", "P13 fine vs closed"),
    ]
    selected = [
        summary_lookup[(family, config, component, "k_le_2")]
        for family, config, component, _ in selections
    ]
    labels = [label for _, _, _, label in selections]
    values = [row["max_abs_relative"] for row in selected]
    positions = np.arange(len(values))
    ax.barh(positions, values, color="tab:blue", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"max relative difference for $k\leq2\ h/{\rm Mpc}$")
    ax.set_title(r"numerical sensitivity of $P_{\delta\theta}$ components")
    ax.grid(alpha=0.25, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "dtheta_numerical_sensitivity.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--redshift", type=float, default=3.0)
    parser.add_argument("--k-values", type=float, nargs="+", default=DEFAULT_K)
    parser.add_argument("--q-maxes", type=float, nargs="+", default=DEFAULT_QMAX)
    parser.add_argument(
        "--linear-source", choices=("auto", "bundle", "class"), default="auto"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    k_values = np.unique(np.asarray(args.k_values, dtype=float))
    q_maxes = np.unique(np.asarray(args.q_maxes, dtype=float))
    if np.any(k_values <= 0) or np.any(q_maxes <= 0):
        raise ValueError("k values and q-max values must be positive.")

    bundle = load_theory(args.theory)
    numerics = dict(bundle.metadata["numerics"])
    required_k_max = float(np.max(q_maxes) + np.max(k_values))
    linear_k, linear_p, linear_metadata = _linear_input(
        bundle, args.redshift, required_k_max, args.linear_source
    )
    print(
        f"linear source: {linear_metadata['kind']}; support to "
        f"{linear_k[-1]:.6g} h/Mpc",
        flush=True,
    )

    tasks, p22_prod, p13_prod = _build_tasks(
        linear_k, linear_p, k_values, q_maxes, numerics
    )
    print(f"running {len(tasks)} tasks with {args.jobs} worker(s)", flush=True)
    rows = _complete_rows(_run_tasks(tasks, args.jobs), k_values, q_maxes)
    summaries = _summaries(rows, k_values, q_maxes)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "dtheta_pointwise.csv", rows)
    _write_csv(output_dir / "dtheta_convergence_summary.csv", summaries)

    lookup = _by_key(rows)
    base = [
        lookup[("qmax", f"qmax_{q_maxes[0]:g}", float(k))] for k in k_values
    ]
    analytic_summary = [
        row
        for row in summaries
        if row["family"] == "analytic_p13" and row["range"] == "k_le_2"
    ][0]
    qmax_summary = [
        row
        for row in summaries
        if row["family"] == "qmax"
        and row["config"] == f"qmax_{q_maxes[0]:g}"
        and row["component"] == "loop_sum_closed_p13"
        and row["range"] == "k_le_2"
    ][0]
    p22_summary = [
        row
        for row in summaries
        if row["family"] == "p22_numerics"
        and row["config"] == "p22_production"
        and row["component"] == "p22_dtheta"
        and row["range"] == "k_le_2"
    ][0]
    report = {
        "theory": str(args.theory.resolve()),
        "bundle_digest": bundle.metadata["bundle_digest"],
        "model": bundle.metadata["model"]["name"],
        "redshift": args.redshift,
        "k_values_hmpc": k_values,
        "q_maxes_hmpc": q_maxes,
        "linear_input": linear_metadata,
        "production_numerics": {"p22": p22_prod, "p13": p13_prod},
        "diagnostics": {
            "p13_direct_vs_closed_max_abs_relative_k_le_2": analytic_summary[
                "max_abs_relative"
            ],
            "closed_p13_loop_sum_qmax_first_vs_last_max_abs_relative_k_le_2": qmax_summary[
                "max_abs_relative"
            ],
            "p22_production_vs_fine_max_abs_relative_k_le_2": p22_summary[
                "max_abs_relative"
            ],
            "cancellation_factor_qmax_first_k_le_2_median": float(
                np.median(
                    [
                        row["cancellation_factor"]
                        for row in base
                        if row["k_hmpc"] <= 2.0
                    ]
                )
            ),
            "cancellation_factor_qmax_first_all_k_max": float(
                np.max([row["cancellation_factor"] for row in base])
            ),
        },
        "note": (
            "No pass/fail threshold is imposed. Inspect the numerical differences, "
            "the analytic P13 comparison, and the cancellation amplification."
        ),
    }
    (output_dir / "dtheta_validation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    if not args.no_plots:
        _plot(rows, summaries, k_values, q_maxes, output_dir)
    print(f"saved diagnostics: {output_dir.resolve()}")
    print(json.dumps(report["diagnostics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
