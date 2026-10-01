"""Controlled CLASS/SPT precision comparisons submitted as one Slurm array."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from campaigns.accdm.build_grid import (
    _atomic_json, _digest, _expand_points, _validate_base_model, build_campaign, point_id,
)
from cluster.campaign_manager import Campaign, _qsub_variables, query_queue
from lyalpha_pt.class_runtime import class_runtime

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "campaigns/accdm/base_models/base_model_refactor_birth.json"
POINTS = ROOT / "campaigns/accdm/grid_specs/refactor_precision_points.json"
VARIANTS = {
    "baseline": ({}, "production"),
    "q100": ({"accdm_q_bins_per_decade": 100.0}, "production"),
    "background80k": ({"background_Nloga": 80000}, "production"),
    "birthtol1e8": ({"accdm_q_number_tol": 1e-8}, "production"),
    "joint": ({"background_Nloga": 80000, "accdm_q_bins_per_decade": 100.0,
               "accdm_q_number_tol": 1e-8}, "production"),
    "joint_spt": ({"background_Nloga": 80000, "accdm_q_bins_per_decade": 100.0,
                   "accdm_q_number_tol": 1e-8}, "precision"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(args):
    if args.cpus < 1 or args.cpu_budget < args.cpus:
        raise ValueError("Require cpu-budget >= cpus >= 1.")
    raw_points = read_json(args.points)["points"]
    normalized = _expand_points({"points": raw_points})
    if not normalized or not all(p["active"] for p in normalized):
        raise ValueError("Provide at least one point; all selected points must be active.")
    points = [dict(label=p.get("label", n["point_id"]),
                   log10m_acc=n["log10m_acc"], log10f_acc=n["log10f_acc"],
                   point_id=n["point_id"]) for p, n in zip(raw_points, normalized)]
    runtime = class_runtime(source_dir=args.class_source)
    base = read_json(BASE)
    _validate_base_model(base)
    suite = args.suite.expanduser().resolve()
    variants = list(dict.fromkeys(args.variants))
    config = {
        "schema_version": 1, "points": points, "variant_order": variants, "variants": {
            name: {"class_overrides": VARIANTS[name][0], "spt_quality":
                   args.software_test_quality or VARIANTS[name][1]} for name in variants
        },
        "base_model": base, "runtime": runtime,
        "software_test_only": bool(args.software_test_quality),
        "resources": {"cpus": args.cpus, "cpu_budget": args.cpu_budget,
                      "array_concurrency": args.cpu_budget // args.cpus,
                      "memory": args.memory, "walltime": args.walltime,
                      "account": args.account, "partition": args.partition},
        "fit": {"mode": "both", "seeds": [12345, 23456, 34567, 45678, 98762],
                "de_maxiter": 300, "de_popsize": 20, "expanded_counterterm": True,
                "no_cutoff_continuation": True, "max_attempts": 1},
    }
    identity = _digest(config)
    destination = suite / "suite.json"
    if destination.exists() and read_json(destination)["input_hash"] != identity:
        raise ValueError("Suite settings changed; choose a new --suite directory.")
    suite.mkdir(parents=True, exist_ok=True)
    (suite / "slurm").mkdir(exist_ok=True)
    cases = []
    for name, variant in config["variants"].items():
        model = json.loads(json.dumps(base))
        model["class_params"].update(variant["class_overrides"])
        base_path = suite / "inputs" / f"{name}_base.json"
        spec_path = suite / "inputs" / f"{name}_grid.json"
        _atomic_json(base_path, model)
        cluster = {
            "scheduler": "slurm", "account": args.account, "partition": args.partition,
            "theory_python": sys.executable, "fit_python": sys.executable,
            "theory_ncpus": args.cpus, "theory_threads": args.cpus,
            "fit_ncpus": 1, "theory_mem": args.memory, "fit_mem": args.memory,
            "theory_walltime": args.walltime, "fit_walltime": args.walltime,
            "class_source_dir": runtime["declared_source"]["directory"],
            "class_source_commit": runtime["declared_source"]["commit"],
            "class_wrapper_sha256": runtime["classy_sha256"],
        }
        spec = {"campaign_id": f"{suite.name}_{name}", "points": points,
                "description": "Isolated precision test; fixed cosmology per point.",
                "theory": {"quality": variant["spt_quality"], "max_attempts": 1},
                "fit": config["fit"], "cluster": cluster}
        _atomic_json(spec_path, spec)
        campaign_dir = suite / name
        build_campaign(base_path, spec_path, campaign_dir)
        for point in points:
            cases.append({"task_id": len(cases), "variant": name,
                          "campaign": name, **point})
    config.update(input_hash=identity, cases=cases)
    _atomic_json(destination, config)
    print(f"Prepared {len(points)} points x {len(variants)} variants = {len(cases)} tasks")
    print(f"At most {min(len(cases), config['resources']['array_concurrency'])} tasks, "
          f"{args.cpus} CPU(s) each; CPU budget {args.cpu_budget}")
    print(f"Suite: {suite}\nNo jobs submitted.")
    return config


def case_context(suite, case):
    campaign = Campaign.load(suite / case["campaign"], ROOT)
    point = next(p for p in campaign.points if p.point_id == case["point_id"])
    return campaign, point


def case_state(campaign, point):
    for stage in ("theory", "fit"):
        if campaign.status_path(point, stage, "failed").exists():
            return f"{stage}_failed"
        if not (campaign.status_path(point, stage, "done").exists()
                and campaign.output_path(point, stage).exists()):
            return f"{stage}_incomplete"
    return "complete"


def run_point(suite, task_id):
    config = read_json(suite / "suite.json")
    if task_id < 0 or task_id >= len(config["cases"]):
        raise ValueError("Task index outside the prepared suite.")
    case = config["cases"][task_id]
    campaign, point = case_context(suite, case)
    cpus = config["resources"]["cpus"]
    if int(os.environ.get("SLURM_CPUS_PER_TASK", cpus)) < cpus:
        raise ValueError("Slurm CPU allocation is smaller than the prepared request.")
    for stage in ("theory", "fit"):
        env = dict(os.environ, **_qsub_variables(campaign, point, stage))
        started = time.monotonic()
        result = subprocess.run(["bash", str(ROOT / "cluster" / f"{stage}_point.slurm")],
                                env=env, check=False)
        _atomic_json(suite / "timings" / f"{task_id}_{stage}.json", {
            "case": case, "stage": stage, "seconds": time.monotonic() - started,
            "exit_code": result.returncode, "allocated_cpus": cpus,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        if result.returncode:
            return result.returncode
    return 0


def submit(suite, *, dry_run=False):
    config = read_json(suite / "suite.json")
    runtime = config["runtime"]
    class_runtime(source_dir=runtime["declared_source"]["directory"],
                  expected_commit=runtime["declared_source"]["commit"],
                  expected_wrapper=runtime["classy_sha256"])
    if str(Path(sys.executable).absolute()) != str(Path(runtime["python"]).absolute()):
        raise ValueError("Activate the Python environment used when building this suite.")
    # One submission at a time; a retry must not duplicate live array tasks.
    with (suite / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = suite / "submission.json"
        if record_path.exists() and not dry_run:
            old_id = read_json(record_path)["job_id"]
            snapshot = query_queue(case_context(suite, config["cases"][0])[0])
            if any(j == old_id or j.startswith(old_id + "_") for j in snapshot.active_job_ids):
                raise RuntimeError(f"Array {old_id} is still active. No duplicate submission.")
        todo = [c["task_id"] for c in config["cases"]
                if case_state(*case_context(suite, c)) != "complete"]
        if not todo:
            print("All tasks complete; nothing to submit.")
            return None
        r = config["resources"]
        command = ["sbatch", "--parsable", "--export=ALL", "--nodes=1", "--ntasks=1",
                   f"--array={','.join(map(str,todo))}%{r['array_concurrency']}",
                   f"--cpus-per-task={r['cpus']}", f"--mem={r['memory']}",
                   f"--time={r['walltime']}", f"--account={r['account']}",
                   f"--partition={r['partition']}", "--job-name=lya_precision",
                   f"--chdir={ROOT}", f"--output={suite}/slurm/%A_%a.out",
                   f"--error={suite}/slurm/%A_%a.err", str(ROOT / "cluster/precision_point.slurm")]
        variables = {"PROJECT_DIR": str(ROOT), "SUITE_DIR": str(suite),
                     "PYTHON_BIN": runtime["python"]}
        print(shlex.join(["env", *[f"{k}={v}" for k,v in variables.items()], *command]))
        if dry_run:
            return command
        result = subprocess.run(command, env=dict(os.environ, **variables), text=True,
                                capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
        match = re.fullmatch(r"(\d+)(?:;[^\s]+)?", result.stdout.strip())
        if not match:
            raise RuntimeError(f"Unexpected sbatch output: {result.stdout!r}; check squeue before retrying.")
        record = {"job_id": match.group(1), "task_ids": todo, "command": command}
        _atomic_json(record_path, record)
        _atomic_json(suite / "submissions" / f"{record['job_id']}.json", record)
        print(f"Submitted array {record['job_id']}; no running controller/tmux is required.")
        return command


def checked_result(suite, case):
    from lyalpha_pt.theory import LoopNumerics, load_theory
    campaign, point = case_context(suite, case)
    bundle = load_theory(campaign.output_path(point, "theory"))
    if bundle.metadata["model"] != read_json(campaign.model_path(point)):
        raise ValueError("Model differs from the immutable campaign input.")
    if bundle.metadata["numerics"] != asdict(LoopNumerics.for_quality(campaign.config["theory"]["quality"])):
        raise ValueError("SPT precision differs from the prepared campaign.")
    runtime = read_json(suite / "suite.json")["runtime"]
    stored = bundle.metadata.get("class_runtime", {})
    if (stored.get("classy_sha256") != runtime["classy_sha256"] or
            stored.get("declared_source") != runtime["declared_source"]):
        raise ValueError("CLASS runtime differs from the prepared suite.")
    payload = read_json(campaign.output_path(point, "fit"))
    for mode in ("linear", "one_loop"):
        result = payload["fits"][mode]
        if result["theory_digest"] != bundle.metadata["bundle_digest"]:
            raise ValueError("Fit uses a different theory bundle.")
        if not math.isfinite(float(result["chi2"])):
            raise ValueError("Non-finite chi2 in fit.")
    return bundle, payload


def report(suite):
    """Compare each variant to the same cosmological point, never to a shifted map minimum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from lyalpha_pt.data import load_dr12
    from lyalpha_pt.fit import EffectiveModelFit, PARAMETER_NAMES
    from scipy.interpolate import InterpolatedUnivariateSpline

    config = read_json(suite / "suite.json")
    output = suite / "analysis"
    output.mkdir(exist_ok=True)
    loaded, rows = {}, []
    for case in config["cases"]:
        campaign, point = case_context(suite, case)
        row = dict(case, status=case_state(campaign, point))
        if row["status"] == "complete":
            try:
                bundle, fit = checked_result(suite, case)
                loaded[(case["point_id"], case["variant"])] = bundle, fit
                loop = fit["fits"]["one_loop"]
                row.update(chi2_linear=fit["fits"]["linear"]["chi2"],
                           chi2_one_loop=loop["chi2"], optimizer_success=loop["optimizer_success"],
                           minimum_bound_distance=min(d["relative_distance_to_nearest_bound"]
                                                      for d in loop["boundary_diagnostics"]))
                row.update({f"nuisance_{k}": v for k,v in loop["parameters"].items()})
            except (ValueError, KeyError, OSError) as error:
                row.update(status="invalid", error=str(error))
        rows.append(row)
    dataset = load_dr12(ROOT / "data")
    for point in config["points"]:
        key = point["point_id"]
        reference = loaded.get((key, "baseline"))
        if reference is None:
            continue
        ref, ref_fit = reference
        ref_fitter = EffectiveModelFit(dataset, ref, mode="one_loop", k_uv_cut=20.0,
                                       covariance_mode="paper_diag")
        theta_ref = np.array([ref_fit["fits"]["one_loop"]["parameters"][n] for n in PARAMETER_NAMES])
        ref_p1d = ref_fitter.model(theta_ref)
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for row in (r for r in rows if r["point_id"] == key and r["status"] == "complete"):
            name = row["variant"]
            bundle, fit = loaded[(key, name)]
            if not np.array_equal(bundle.z, ref.z):
                raise ValueError("Compared bundles have different redshift grids.")
            fitter = EffectiveModelFit(dataset, bundle, mode="one_loop", k_uv_cut=20.0,
                                       covariance_mode="paper_diag")
            theta = np.array([fit["fits"]["one_loop"]["parameters"][n] for n in PARAMETER_NAMES])
            recomputed = fitter.chi2(theta)
            if not np.isclose(recomputed, row["chi2_one_loop"], rtol=1e-8, atol=1e-5):
                raise ValueError(f"Fit chi2 verification failed: {key} {name}")
            # alpha_ct is normalized by a theory-dependent scale: preserve the physical amplitude.
            transferred = theta_ref.copy()
            transferred[4] *= ref_fitter.i0_scale / fitter.i0_scale
            fixed, profiled = fitter.model(transferred), fitter.model(theta)
            row["delta_chi2_vs_baseline"] = row["chi2_one_loop"] - ref_fit["fits"]["one_loop"]["chi2"]
            row["chi2_at_fixed_baseline_nuisance"] = fitter.chi2(transferred)
            row["p1d_fixed_max_abs_pct"] = float(np.max(np.abs(100*(fixed/ref_p1d-1))))
            row["p1d_profiled_max_abs_pct"] = float(np.max(np.abs(100*(profiled/ref_p1d-1))))
            k = np.geomspace(max(ref.k_input[0],bundle.k_input[0]),
                             min(ref.k_input[-1],bundle.k_input[-1]), 900)
            kt = np.geomspace(max(ref.k_loop[0],bundle.k_loop[0]),
                              min(ref.metadata['numerics']['k_trust'],bundle.metadata['numerics']['k_trust']), 400)
            linear, loops = [], []
            for iz in range(len(ref.z)):
                rp = np.exp(InterpolatedUnivariateSpline(np.log(ref.k_input),np.log(ref.p_total_input[iz]),k=3)(np.log(k)))
                bp = np.exp(InterpolatedUnivariateSpline(np.log(bundle.k_input),np.log(bundle.p_total_input[iz]),k=3)(np.log(k)))
                linear.append(100*(bp/rp-1))
                channels=[]
                for ch in range(3):
                    rs=InterpolatedUnivariateSpline(np.log(ref.k_loop),ref.channels_one_loop[iz,:,ch],k=3)(np.log(kt))
                    bs=InterpolatedUnivariateSpline(np.log(bundle.k_loop),bundle.channels_one_loop[iz,:,ch],k=3)(np.log(kt))
                    channels.append(100*(bs-rs)/np.maximum(np.abs(rs),1e-12*np.max(np.abs(rs))))
                loops.append(channels)
            row["linear_full_input_max_abs_pct"] = float(np.max(np.abs(linear)))
            row["one_loop_trusted_max_abs_pct"] = float(np.max(np.abs(loops)))
            if name != "baseline":
                axes[0,0].plot(k,np.max(np.abs(linear),axis=0),label=name)
                axes[0,1].plot(kt,np.max(np.abs(loops),axis=(0,1)),label=name)
                mask=np.isclose(dataset.z,3.6)
                axes[1,0].plot(dataset.k_velocity[mask],100*(fixed[mask]/ref_p1d[mask]-1),label=name)
                axes[1,1].plot(dataset.k_velocity[mask],100*(profiled[mask]/ref_p1d[mask]-1),label=name)
        for axis, title in zip(axes.flat,["Linear P(k): max over redshifts", "One-loop: max over redshifts/channels",
                                         "P1D: fixed physical nuisance, z=3.6", "P1D: refitted nuisance, z=3.6"]):
            axis.set_title(title)
            axis.set_ylabel("Difference from baseline [%]")
            axis.grid(alpha=.25)
        for axis in axes[0]:
            axis.set_xscale("log")
            axis.set_xlabel("k [h/Mpc]")
        for axis in axes[1]:
            axis.set_xlabel("k [s/km]")
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:
            figure.legend(handles,labels,loc="lower center",ncol=3)
        figure.suptitle(f"{point['label']}: log10(m)={point['log10m_acc']:.6g}, log10(f)={point['log10f_acc']:.6g}")
        figure.tight_layout(rect=(0,.065,1,.95))
        figure.savefig(output/f"{key}_precision.png",dpi=170)
        plt.close(figure)
    for row in rows:
        if row["variant"] == "joint_spt" and row["status"] == "complete":
            joint=loaded.get((row["point_id"],"joint"))
            if joint is not None:
                row["delta_chi2_spt_vs_joint"] = row["chi2_one_loop"]-joint[1]["fits"]["one_loop"]["chi2"]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/"precision_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if loaded:
        figure,axes=plt.subplots(1,2,figsize=(13,5))
        names=list(config['variants'])
        for point in config['points']:
            lookup={r['variant']:r for r in rows if r['point_id']==point['point_id']}
            axes[0].plot(range(len(names)),[lookup[n].get('chi2_one_loop',np.nan) for n in names],
                         marker='o',label=point['label'])
            axes[1].plot(range(len(names)),[lookup[n].get('delta_chi2_vs_baseline',np.nan) for n in names],
                         marker='o',label=point['label'])
        axes[0].set_ylabel('One-loop chi2 after nuisance fit')
        axes[1].set_ylabel('chi2 - baseline chi2 at the same point')
        for axis in axes:
            axis.set_xticks(range(len(names)),names,rotation=35,ha='right')
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output/'chi2_precision.png',dpi=170)
        plt.close(figure)
    _atomic_json(output/"report_metadata.json", {"input_hash":config["input_hash"],
        "status_counts":dict(Counter(r["status"] for r in rows)),
        "delta_chi2_reference":"same point at baseline precision; no map-wise renormalization",
        "software_test_only": config["software_test_only"],
        "spt_comparison":"joint_spt versus joint changes SPT only, including its integration range",
        "convergence_certified":False})
    print(f"Report: {output}\nStates: {dict(Counter(r['status'] for r in rows))}")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,default=ROOT/"runs/accdm/refactor_precision_v1")
    commands=parser.add_subparsers(dest="command",required=True)
    build_parser=commands.add_parser("build")
    build_parser.add_argument("--class-source",type=Path,required=True)
    build_parser.add_argument("--account",required=True)
    build_parser.add_argument("--points",type=Path,default=POINTS)
    build_parser.add_argument("--variants",nargs="+",choices=tuple(VARIANTS),default=list(VARIANTS))
    build_parser.add_argument("--cpus",type=int,default=1)
    build_parser.add_argument("--cpu-budget",type=int,default=40)
    build_parser.add_argument("--memory",default="8G")
    build_parser.add_argument("--partition",default="standard")
    build_parser.add_argument("--walltime",default="24:00:00")
    build_parser.add_argument("--software-test-quality",choices=("smoke",),help="Software test only; not precision validation.")
    submit_parser=commands.add_parser("submit")
    submit_parser.add_argument("--dry-run",action="store_true")
    commands.add_parser("status")
    commands.add_parser("report")
    run_parser=commands.add_parser("run-point")
    run_parser.add_argument("--task-id",type=int,required=True)
    args=parser.parse_args()
    args.suite=args.suite.expanduser().resolve()
    if args.command=="build": build(args)
    elif args.command=="submit": submit(args.suite,dry_run=args.dry_run)
    elif args.command=="run-point": return run_point(args.suite,args.task_id)
    elif args.command=="report": report(args.suite)
    else:
        config=read_json(args.suite/"suite.json")
        print(json.dumps(dict(Counter(case_state(*case_context(args.suite,c)) for c in config["cases"])),indent=2))
        if (args.suite/"submission.json").exists():
            print("Array job:",read_json(args.suite/"submission.json")["job_id"])
        print("This status uses files; use squeue/sacct for live scheduler state.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
