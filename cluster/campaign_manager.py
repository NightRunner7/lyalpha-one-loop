#!/usr/bin/env python3
"""Submit, monitor and resume model-independent Ly-alpha PBS/Slurm campaigns."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("theory", "fit")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Point:
    index: int
    point_id: str
    active: bool
    model_json: str
    model_hash: str
    values: Mapping[str, str]


@dataclass(frozen=True)
class QueueSnapshot:
    active_job_ids: frozenset[str]
    total_active: int


@dataclass(frozen=True)
class Campaign:
    root: Path
    project_root: Path
    config: Mapping[str, Any]
    points: tuple[Point, ...]

    @classmethod
    def load(cls, root: Path, project_root: Path = PROJECT_ROOT) -> "Campaign":
        root = root.resolve()
        project_root = project_root.resolve()
        config = _load_json(root / "campaign.json")
        manifest_name = str(config.get("manifest", "manifest.csv"))
        manifest_path = root / manifest_name
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Campaign manifest is empty: {manifest_path}")

        required = {"index", "point_id", "active", "model_json", "model_hash"}
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        points: list[Point] = []
        seen_indices: set[int] = set()
        seen_ids: set[str] = set()
        for row in rows:
            index = int(row["index"])
            point_id = row["point_id"]
            if index in seen_indices or point_id in seen_ids:
                raise ValueError("Manifest has duplicate indices or point IDs.")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", point_id):
                raise ValueError(f"Unsafe point_id in manifest: {point_id!r}")
            seen_indices.add(index)
            seen_ids.add(point_id)
            model_path = root / row["model_json"]
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing model JSON for {point_id}: {model_path}")
            points.append(
                Point(
                    index=index,
                    point_id=point_id,
                    active=_truthy(row["active"]),
                    model_json=row["model_json"],
                    model_hash=row["model_hash"],
                    values=dict(row),
                )
            )
        points.sort(key=lambda point: point.index)
        return cls(root=root, project_root=project_root, config=config, points=tuple(points))

    @property
    def campaign_id(self) -> str:
        return str(self.config["campaign_id"])

    def output_path(self, point: Point, stage: str) -> Path:
        if stage == "theory":
            return self.root / "theory" / f"{point.point_id}.npz"
        if stage == "fit":
            return self.root / "fits" / f"{point.point_id}.json"
        raise ValueError(stage)

    def status_path(self, point: Point, stage: str, kind: str) -> Path:
        return self.root / "status" / stage / f"{point.point_id}.{kind}.json"

    def job_record_path(self, point: Point, stage: str) -> Path:
        return self.root / "jobs" / stage / f"{point.point_id}.json"

    def model_path(self, point: Point) -> Path:
        return (self.root / point.model_json).resolve()

    def data_path(self) -> Path:
        value = Path(str(self.config.get("data_dir", "data")))
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()


def _load_job_record(campaign: Campaign, point: Point, stage: str) -> dict[str, Any] | None:
    path = campaign.job_record_path(point, stage)
    return _load_json(path) if path.exists() else None


def _job_id_key(job_id: str) -> str:
    return str(job_id).strip().split(".", 1)[0]


def _record_is_active(record: Mapping[str, Any] | None, snapshot: QueueSnapshot | None) -> bool:
    if record is None or snapshot is None:
        return False
    wanted = _job_id_key(str(record.get("job_id", "")))
    return bool(wanted) and any(_job_id_key(job_id) == wanted for job_id in snapshot.active_job_ids)


def point_state(
    campaign: Campaign,
    point: Point,
    stage: str,
    snapshot: QueueSnapshot | None,
) -> str:
    """Return a filesystem-plus-scheduler state for one point and stage."""

    if not point.active:
        return "inactive"
    output = campaign.output_path(point, stage)
    done = campaign.status_path(point, stage, "done")
    failed = campaign.status_path(point, stage, "failed")
    record = _load_job_record(campaign, point, stage)

    if done.exists() and output.exists():
        return "complete"
    if _record_is_active(record, snapshot):
        return "active"
    if failed.exists():
        return "failed"
    if stage == "fit":
        theory_state = point_state(campaign, point, "theory", snapshot)
        if theory_state == "failed" or theory_state == "exhausted":
            return "blocked_theory_failed"
        if theory_state != "complete":
            return "waiting_theory"
    if record is not None and snapshot is None:
        return "submitted_unknown"
    max_attempts = int(campaign.config[stage].get("max_attempts", 3))
    if record is not None and int(record.get("attempt", 0)) >= max_attempts:
        return "exhausted"
    return "pending"


def _command_words(value: Any, default: str) -> list[str]:
    if value is None:
        return [default]
    if isinstance(value, list):
        return [str(item) for item in value]
    words = shlex.split(str(value))
    return words or [default]


def _scheduler(campaign: Campaign) -> str:
    scheduler = str(campaign.config.get("cluster", {}).get("scheduler", "pbs")).lower()
    if scheduler not in {"pbs", "slurm"}:
        raise ValueError(f"Unsupported cluster.scheduler: {scheduler!r}")
    return scheduler


def query_queue(campaign: Campaign) -> QueueSnapshot:
    cluster = campaign.config.get("cluster", {})
    if _scheduler(campaign) == "slurm":
        command = _command_words(cluster.get("squeue_command"), "squeue")
        user = str(cluster.get("slurm_user", os.environ.get("USER") or getpass.getuser()))
        result = subprocess.run(
            [*command, "--noheader", "--array", "--user", user,
             "--states=all", "--format=%i|%T"],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"squeue failed with exit code {result.returncode}: {result.stderr.strip()}"
            )
        terminal = {
            "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
        }
        job_ids: set[str] = set()
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            fields = [value.strip() for value in line.split("|")]
            if len(fields) != 2 or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?(?:\+[0-9]+)?", fields[0]) or not fields[1]:
                raise RuntimeError(f"Cannot parse squeue output; refusing to submit: {line!r}")
            # Unknown/nonterminal states stay active: never resubmit a held,
            # suspended, requeued or still-completing job.
            if fields[1] not in terminal:
                job_ids.add(fields[0])
        return QueueSnapshot(frozenset(job_ids), len(job_ids))
    command = _command_words(cluster.get("qstat_command"), "qstat")
    user = str(cluster.get("pbs_user", os.environ.get("USER") or getpass.getuser()))
    result = subprocess.run(
        [*command, "-u", user],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"qstat failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    job_ids: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and re.match(r"^\d", fields[0]):
            job_ids.add(fields[0])
    return QueueSnapshot(frozenset(job_ids), len(job_ids))


def summarize(campaign: Campaign, snapshot: QueueSnapshot | None) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in STAGES:
        states = Counter(
            point_state(campaign, point, stage, snapshot) for point in campaign.points
        )
        stages[stage] = {
            "counts": dict(sorted(states.items())),
            "failed_points": [
                point.point_id
                for point in campaign.points
                if point_state(campaign, point, stage, snapshot) in {"failed", "exhausted"}
            ],
        }
    return {
        "campaign_id": campaign.campaign_id,
        "campaign_dir": str(campaign.root),
        "queue_total_active": None if snapshot is None else snapshot.total_active,
        "stages": stages,
    }


def print_summary(campaign: Campaign, snapshot: QueueSnapshot | None) -> dict[str, Any]:
    summary = summarize(campaign, snapshot)
    queue_label = "offline" if snapshot is None else str(snapshot.total_active)
    print(f"campaign: {campaign.campaign_id}")
    print(f"directory: {campaign.root}")
    queue_command = "squeue" if _scheduler(campaign) == "slurm" else "qstat"
    print(f"active jobs visible to {queue_command}: {queue_label}")
    for stage in STAGES:
        counts = summary["stages"][stage]["counts"]
        rendered = ", ".join(f"{name}={value}" for name, value in counts.items())
        print(f"{stage:6s}: {rendered}")
        failed_points = summary["stages"][stage]["failed_points"]
        if failed_points:
            preview = ", ".join(failed_points[:8])
            suffix = " ..." if len(failed_points) > 8 else ""
            print(f"         needs attention: {preview}{suffix}")
    return summary


def _validate_qsub_variables(variables: Mapping[str, str]) -> None:
    for name, value in variables.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"Unsafe PBS variable name: {name!r}")
        if any(character in value for character in (",", "\n", "\r")):
            raise ValueError(
                f"PBS variable {name} contains a comma or newline, which qsub -v cannot encode."
            )


def _job_name(campaign: Campaign, point: Point, stage: str) -> str:
    digest = hashlib.sha256(campaign.campaign_id.encode("utf-8")).hexdigest()[:4]
    stage_letter = "t" if stage == "theory" else "f"
    return f"lya{stage_letter}{digest}{point.index:06d}"[:15]


def _qsub_variables(campaign: Campaign, point: Point, stage: str) -> dict[str, str]:
    cluster = campaign.config["cluster"]
    variables = {
        "PROJECT_DIR": str(campaign.project_root),
        "CAMPAIGN_DIR": str(campaign.root),
        "POINT_ID": point.point_id,
        "CONDA_EXE": str(cluster.get("conda_exe", "/opt/anaconda3/bin/conda")),
        "PYTHON_BIN": "",
    }
    if stage == "theory":
        variables.update(
            {
                "MODEL_JSON": str(campaign.model_path(point)),
                "OUTPUT_FILE": str(campaign.output_path(point, stage)),
                "CHECKPOINT_DIR": str(campaign.root / "checkpoints" / point.point_id),
                "QUALITY": str(campaign.config["theory"].get("quality", "production")),
                "ENV_NAME": str(cluster.get("theory_env", "class_decays")),
                "THEORY_THREADS": str(
                    int(
                        cluster.get(
                            "theory_threads",
                            cluster.get("theory_ncpus", 1),
                        )
                    )
                ),
            }
        )
        python_bin = str(cluster.get("theory_python", ""))
    else:
        fit = campaign.config["fit"]
        fit_strategy = str(fit.get("strategy", "standard"))
        if fit_strategy not in {"standard", "neighbor_refit"}:
            raise ValueError(f"Unsupported fit strategy: {fit_strategy!r}")
        source_directories = [
            Path(str(value)) for value in fit.get("source_fit_dirs", [])
        ]
        if fit_strategy == "neighbor_refit" and not source_directories:
            raise ValueError(
                "fit.strategy='neighbor_refit' requires fit.source_fit_dirs."
            )
        resolved_sources = [
            path.resolve() if path.is_absolute() else (campaign.root / path).resolve()
            for path in source_directories
        ]
        if any(":" in str(path) for path in resolved_sources):
            raise ValueError("Fit source paths cannot contain ':'.")
        variables.update(
            {
                "THEORY_FILE": str(campaign.output_path(point, "theory")),
                "OUTPUT_FILE": str(campaign.output_path(point, stage)),
                "DATA_DIR": str(campaign.data_path()),
                "MODE": str(fit.get("mode", "one_loop")),
                "K_UV_CUT": str(fit.get("k_uv_cut", 20.0)),
                "COVARIANCE": str(fit.get("covariance", "paper_diag")),
                "SEEDS": ":".join(str(seed) for seed in fit.get("seeds", [])),
                "EXPANDED_COUNTERTERM": "1" if fit.get("expanded_counterterm") else "0",
                "FULL_SIX_DIMENSIONAL": "1" if fit.get("full_six_dimensional") else "0",
                "NO_CUTOFF_CONTINUATION": "1" if fit.get("no_cutoff_continuation") else "0",
                "FIT_STRATEGY": fit_strategy,
                "FIT_SOURCE_DIRS": ":".join(str(path) for path in resolved_sources),
                "DE_MAXITER": str(fit.get("de_maxiter", 300)),
                "DE_POPSIZE": str(fit.get("de_popsize", 20)),
                "ENV_NAME": str(cluster.get("fit_env", "class_decays")),
            }
        )
        python_bin = str(cluster.get("fit_python", ""))
    if python_bin:
        variables["PYTHON_BIN"] = python_bin
    if stage == "theory" and cluster.get("class_source_dir"):
        variables["LYA_CLASS_SOURCE_DIR"] = str(cluster["class_source_dir"])
        variables["LYA_CLASS_SOURCE_COMMIT"] = str(cluster.get("class_source_commit", ""))
        variables["LYA_CLASS_WRAPPER_SHA256"] = str(cluster.get("class_wrapper_sha256", ""))
    if _scheduler(campaign) == "pbs":
        _validate_qsub_variables(variables)
    return variables


def _stage_resource_request(campaign: Campaign, stage: str) -> str:
    """Return one validated PBS select request for the selected stage."""

    cluster = campaign.config["cluster"]
    ncpus = int(cluster.get(f"{stage}_ncpus", 1))
    if ncpus < 1:
        raise ValueError(f"cluster.{stage}_ncpus must be positive.")
    default_memory = "8gb" if stage == "theory" else "4gb"
    memory = str(cluster.get(f"{stage}_mem", default_memory)).strip().lower()
    if not re.fullmatch(r"[1-9][0-9]*(?:kb|mb|gb|tb)", memory):
        raise ValueError(
            f"cluster.{stage}_mem must look like '4gb' or '800mb', got {memory!r}."
        )
    if stage == "theory":
        threads = int(cluster.get("theory_threads", ncpus))
        if not 1 <= threads <= ncpus:
            raise ValueError(
                "cluster.theory_threads must be between 1 and theory_ncpus."
            )
    return f"select=1:ncpus={ncpus}:mem={memory}"


def _stage_walltime_request(campaign: Campaign, stage: str) -> str | None:
    """Return an optional validated PBS walltime for the selected stage."""

    cluster = campaign.config["cluster"]
    raw = cluster.get(f"{stage}_walltime")
    if raw is None:
        return None
    walltime = str(raw).strip()
    match = re.fullmatch(r"([0-9]+):([0-5][0-9]):([0-5][0-9])", walltime)
    if match is None or not any(int(value) for value in match.groups()):
        raise ValueError(
            f"cluster.{stage}_walltime must look like '48:00:00' and be positive, "
            f"got {walltime!r}."
        )
    return walltime


def _qsub_command(campaign: Campaign, point: Point, stage: str) -> list[str]:
    cluster = campaign.config["cluster"]
    command = _command_words(cluster.get("qsub_command"), "qsub")
    variables = _qsub_variables(campaign, point, stage)
    log_dir = campaign.root / "logs" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    template = campaign.project_root / "cluster" / f"{stage}_point.pbs"
    if not template.is_file():
        raise FileNotFoundError(f"Missing PBS template: {template}")
    resources = ["-l", _stage_resource_request(campaign, stage)]
    walltime = _stage_walltime_request(campaign, stage)
    if walltime is not None:
        resources.extend(("-l", f"walltime={walltime}"))
    return [
        *command,
        "-v",
        ",".join(f"{name}={value}" for name, value in variables.items()),
        *resources,
        "-N",
        _job_name(campaign, point, stage),
        "-o",
        str(log_dir / f"{point.point_id}.out"),
        "-e",
        str(log_dir / f"{point.point_id}.err"),
        str(template),
    ]


def _sbatch_command(campaign: Campaign, point: Point, stage: str) -> list[str]:
    cluster = campaign.config["cluster"]
    # Share the existing validated one-node CPU/memory/thread contract.
    _stage_resource_request(campaign, stage)
    ncpus = int(cluster.get(f"{stage}_ncpus", 1))
    memory = str(cluster.get(f"{stage}_mem", "8gb" if stage == "theory" else "4gb"))
    memory = memory.strip().upper().removesuffix("B")
    walltime = _stage_walltime_request(campaign, stage) or (
        "24:00:00" if stage == "theory" else "12:00:00"
    )
    template = campaign.project_root / "cluster" / f"{stage}_point.slurm"
    if not template.is_file():
        raise FileNotFoundError(f"Missing Slurm template: {template}")
    log_dir = campaign.root / "logs" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *_command_words(cluster.get("sbatch_command"), "sbatch"),
        "--parsable", "--export=ALL", "--nodes=1", "--ntasks=1",
        f"--cpus-per-task={ncpus}", f"--mem={memory}", f"--time={walltime}",
        f"--job-name={_job_name(campaign, point, stage)}",
        f"--chdir={campaign.project_root}",
        f"--output={log_dir / (point.point_id + '.%j.out')}",
        f"--error={log_dir / (point.point_id + '.%j.err')}",
    ]
    for key in ("partition", "account", "qos"):
        value = str(cluster.get(f"{stage}_{key}", cluster.get(key, ""))).strip()
        if value:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError(f"Invalid Slurm {key}: {value!r}")
            command.append(f"--{key}={value}")
    return [*command, str(template)]


def _submit_one(campaign: Campaign, point: Point, stage: str, dry_run: bool) -> str | None:
    scheduler = _scheduler(campaign)
    command = (_sbatch_command if scheduler == "slurm" else _qsub_command)(campaign, point, stage)
    variables = _qsub_variables(campaign, point, stage)
    if dry_run:
        prefix = ["env", *[f"{key}={value}" for key, value in variables.items()]] if scheduler == "slurm" else []
        print(f"[dry-run] {shlex.join([*prefix, *command])}")
        return None
    kwargs = {}
    if scheduler == "slurm":
        # Values travel in the process environment, not in Slurm's comma-
        # separated --export argument (paths may contain spaces or commas).
        kwargs["env"] = {**os.environ, **variables}
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{scheduler} submission failed for {point.point_id}: {result.stderr.strip()}"
        )
    job_id = result.stdout.strip().split()[0] if result.stdout.strip() else ""
    if scheduler == "slurm":
        if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip()):
            raise RuntimeError(f"sbatch returned an invalid job ID: {result.stdout!r}")
        job_id = job_id.split(";", 1)[0]
    elif not job_id:
        raise RuntimeError(f"qsub returned no job ID for {point.point_id}.")
    old_record = _load_job_record(campaign, point, stage) or {}
    record = {
        "campaign_id": campaign.campaign_id,
        "point_id": point.point_id,
        "stage": stage,
        "job_id": job_id,
        "scheduler": scheduler,
        "job_name": _job_name(campaign, point, stage),
        "attempt": int(old_record.get("attempt", 0)) + 1,
        "submitted_at": _utc_now(),
        "command": command,
        "variables": variables,
    }
    _atomic_json(campaign.job_record_path(point, stage), record)
    print(
        f"submitted {stage:6s} {point.point_id} as {job_id} "
        f"(attempt {record['attempt']})"
    )
    return job_id


def submit_stage(
    campaign: Campaign,
    stage: str,
    *,
    dry_run: bool = False,
    offline: bool = False,
    max_active_override: int | None = None,
    max_points: int | None = None,
) -> int:
    if stage not in STAGES:
        raise ValueError(stage)
    if offline and not dry_run:
        raise ValueError("--offline is only safe together with --dry-run.")
    snapshot = None if offline else query_queue(campaign)
    cluster = campaign.config["cluster"]
    configured_max = int(cluster[f"{stage}_max_active"])
    max_active = configured_max if max_active_override is None else max_active_override
    if max_active < 1:
        raise ValueError("max_active must be positive.")

    campaign_active = sum(
        point_state(campaign, point, stage, snapshot) == "active"
        for point in campaign.points
    )
    if snapshot is None:
        free_slots = max_active - campaign_active
    else:
        max_user_active = int(cluster.get("max_user_active", 10**9))
        free_slots = min(
            max_active - campaign_active,
            max_user_active - snapshot.total_active,
        )
    if max_points is not None:
        free_slots = min(free_slots, max_points)
    free_slots = max(0, free_slots)
    if free_slots == 0:
        print(f"no free {stage} slots")
        return 0

    submitted = 0
    delay = float(cluster.get("submit_delay_seconds", 0.0))
    for point in campaign.points:
        if submitted >= free_slots:
            break
        if point_state(campaign, point, stage, snapshot) != "pending":
            continue
        job_id = _submit_one(campaign, point, stage, dry_run)
        submitted += 1
        if snapshot is not None and job_id:
            snapshot = QueueSnapshot(
                frozenset((*snapshot.active_job_ids, job_id)),
                snapshot.total_active + 1,
            )
        if delay > 0.0 and not dry_run:
            time.sleep(delay)
    print(f"{stage}: submitted/planned {submitted} job(s)")
    return submitted


def retry_failed(
    campaign: Campaign, stage: str, point_ids: Sequence[str] | None = None
) -> int:
    stages = STAGES if stage == "both" else (stage,)
    snapshot = query_queue(campaign)
    selected = set(point_ids or ())
    reset = 0
    for current_stage in stages:
        for point in campaign.points:
            if selected and point.point_id not in selected:
                continue
            state = point_state(campaign, point, current_stage, snapshot)
            if state not in {"failed", "exhausted"}:
                continue
            record = _load_job_record(campaign, point, current_stage)
            if _record_is_active(record, snapshot):
                print(f"skip active job: {current_stage} {point.point_id}")
                continue
            campaign.status_path(point, current_stage, "failed").unlink(missing_ok=True)
            campaign.job_record_path(point, current_stage).unlink(missing_ok=True)
            print(f"reset {current_stage:6s} {point.point_id}")
            reset += 1
    print(f"reset {reset} failed/exhausted stage(s)")
    return reset


def prepare_neighbor_refit(
    campaign: Campaign,
    *,
    archive_label: str,
    additional_source_fit_dirs: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Archive a completed fit pass and open a clean neighbour-refit stage."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", archive_label):
        raise ValueError("archive_label may contain only letters, digits, '.', '_' and '-'.")
    incomplete = [
        point.point_id
        for point in campaign.points
        if point.active
        and not (
            campaign.output_path(point, "fit").is_file()
            and campaign.status_path(point, "fit", "done").is_file()
        )
    ]
    if incomplete:
        preview = ", ".join(incomplete[:8])
        suffix = " ..." if len(incomplete) > 8 else ""
        raise RuntimeError(
            "The current fit pass is not complete; refusing to archive it. "
            f"Incomplete points: {preview}{suffix}"
        )

    fit_archive = campaign.root / f"fits_{archive_label}"
    moves = (
        (campaign.root / "fits", fit_archive),
        (campaign.root / "status" / "fit", campaign.root / "status" / f"fit_{archive_label}"),
        (campaign.root / "jobs" / "fit", campaign.root / "jobs" / f"fit_{archive_label}"),
        (campaign.root / "logs" / "fit", campaign.root / "logs" / f"fit_{archive_label}"),
    )
    if not moves[0][0].is_dir():
        raise FileNotFoundError(f"Missing active fit directory: {moves[0][0]}")
    existing_destinations = [str(destination) for _, destination in moves if destination.exists()]
    if existing_destinations:
        raise FileExistsError(
            "Refusing to overwrite archived fit directories: "
            + ", ".join(existing_destinations)
        )

    additional = [Path(value) for value in additional_source_fit_dirs]
    resolved_additional = [
        path.resolve() if path.is_absolute() else (campaign.root / path).resolve()
        for path in additional
    ]
    missing_sources = [str(path) for path in resolved_additional if not path.is_dir()]
    if missing_sources:
        raise FileNotFoundError(
            "Additional source fit directories do not exist: " + ", ".join(missing_sources)
        )

    for source, destination in moves:
        if source.exists():
            shutil.move(str(source), str(destination))
    for path in (
        campaign.root / "fits",
        campaign.root / "status" / "fit",
        campaign.root / "jobs" / "fit",
        campaign.root / "logs" / "fit",
    ):
        path.mkdir(parents=True, exist_ok=True)

    config_path = campaign.root / "campaign.json"
    config = _load_json(config_path)
    fit = dict(config["fit"])
    sources = [fit_archive.name, *[str(path) for path in additional]]
    fit["strategy"] = "neighbor_refit"
    fit["source_fit_dirs"] = sources
    config["fit"] = fit
    _atomic_json(config_path, config)
    return tuple(
        path.resolve() if path.is_absolute() else (campaign.root / path).resolve()
        for path in map(Path, sources)
    )


def _load_campaign_from_args(args: argparse.Namespace) -> Campaign:
    project_root = Path(args.project_dir) if args.project_dir else PROJECT_ROOT
    return Campaign.load(Path(args.campaign), project_root)


def _cmd_status(args: argparse.Namespace) -> int:
    campaign = _load_campaign_from_args(args)
    snapshot = None if args.offline else query_queue(campaign)
    summary = summarize(campaign, snapshot)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(campaign, snapshot)
    return 0


def _cmd_submit(args: argparse.Namespace, stage: str) -> int:
    campaign = _load_campaign_from_args(args)
    submit_stage(
        campaign,
        stage,
        dry_run=args.dry_run,
        offline=args.offline,
        max_active_override=args.max_active,
        max_points=args.max_points,
    )
    return 0


def _cmd_retry(args: argparse.Namespace) -> int:
    campaign = _load_campaign_from_args(args)
    retry_failed(campaign, args.stage, args.point_id)
    return 0


def _cmd_prepare_neighbor_refit(args: argparse.Namespace) -> int:
    campaign = _load_campaign_from_args(args)
    sources = prepare_neighbor_refit(
        campaign,
        archive_label=args.archive_label,
        additional_source_fit_dirs=args.additional_source_fit_dir or (),
    )
    print("completed fit pass archived; neighbour-refit stage is ready")
    for source in sources:
        print(f"source fits: {source}")
    print(f"new output: {campaign.root / 'fits'}")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    campaign = _load_campaign_from_args(args)
    path = campaign.root / "controller" / "STOP"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"requested_at={_utc_now()}\n", encoding="utf-8")
    print(f"stop requested: {path}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    campaign = _load_campaign_from_args(args)
    path = campaign.root / "controller" / "STOP"
    path.unlink(missing_ok=True)
    print(f"stop request cleared: {path}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    project_root = Path(args.project_dir) if args.project_dir else PROJECT_ROOT
    campaign_root = Path(args.campaign)
    stages = tuple(dict.fromkeys(args.stages))
    print(f"watching stages: {', '.join(stages)}")
    print("create controller/STOP or run the 'stop' command to exit cleanly")
    try:
        while True:
            campaign = Campaign.load(campaign_root, project_root)
            stop_path = campaign.root / "controller" / "STOP"
            if stop_path.exists():
                print(f"stop file found: {stop_path}")
                return 0
            for stage in stages:
                submit_stage(
                    campaign,
                    stage,
                    max_active_override=args.max_active,
                    max_points=args.max_points_per_cycle,
                )
                campaign = Campaign.load(campaign_root, project_root)
            snapshot = query_queue(campaign)
            summary = print_summary(campaign, snapshot)
            if args.exit_when_complete:
                requested_complete = all(
                    summary["stages"][stage]["counts"].get("complete", 0)
                    == sum(point.active for point in campaign.points)
                    for stage in stages
                )
                if requested_complete:
                    print("all requested stages are complete")
                    return 0
                terminal_states = {"complete", "inactive", "failed", "exhausted", "blocked_theory_failed"}
                if all(set(summary["stages"][stage]["counts"]).issubset(terminal_states) for stage in stages):
                    print("campaign stopped with failed/exhausted points; inspect logs and use retry-failed")
                    return 2
            interval = (
                args.poll_seconds
                if args.poll_seconds is not None
                else float(campaign.config["cluster"].get("poll_seconds", 30.0))
            )
            time.sleep(max(1.0, interval))
    except KeyboardInterrupt:
        print("controller interrupted by user")
        return 130


def _add_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Project root containing scripts/, cluster/ and data/. Defaults to this installation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show reconciled campaign state.")
    _add_campaign_arguments(status_parser)
    status_parser.add_argument("--offline", action="store_true")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(handler=_cmd_status)

    for command, stage in (("submit-theory", "theory"), ("submit-fits", "fit")):
        submit_parser = subparsers.add_parser(command)
        _add_campaign_arguments(submit_parser)
        submit_parser.add_argument("--dry-run", action="store_true")
        submit_parser.add_argument("--offline", action="store_true")
        submit_parser.add_argument("--max-active", type=int)
        submit_parser.add_argument("--max-points", type=int)
        submit_parser.set_defaults(handler=lambda args, selected=stage: _cmd_submit(args, selected))

    retry_parser = subparsers.add_parser("retry-failed")
    _add_campaign_arguments(retry_parser)
    retry_parser.add_argument("--stage", choices=(*STAGES, "both"), default="both")
    retry_parser.add_argument("--point-id", action="append")
    retry_parser.set_defaults(handler=_cmd_retry)

    prepare_parser = subparsers.add_parser(
        "prepare-neighbor-refit",
        help="Archive a completed fit pass and configure a neighbour-seeded refit.",
    )
    _add_campaign_arguments(prepare_parser)
    prepare_parser.add_argument("--archive-label", default="direct_k20")
    prepare_parser.add_argument("--additional-source-fit-dir", action="append")
    prepare_parser.set_defaults(handler=_cmd_prepare_neighbor_refit)

    watch_parser = subparsers.add_parser("watch")
    _add_campaign_arguments(watch_parser)
    watch_parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    watch_parser.add_argument("--max-active", type=int)
    watch_parser.add_argument("--max-points-per-cycle", type=int)
    watch_parser.add_argument("--poll-seconds", type=float)
    watch_parser.add_argument("--exit-when-complete", action="store_true")
    watch_parser.set_defaults(handler=_cmd_watch)

    stop_parser = subparsers.add_parser("stop")
    _add_campaign_arguments(stop_parser)
    stop_parser.set_defaults(handler=_cmd_stop)

    resume_parser = subparsers.add_parser("resume")
    _add_campaign_arguments(resume_parser)
    resume_parser.set_defaults(handler=_cmd_resume)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
