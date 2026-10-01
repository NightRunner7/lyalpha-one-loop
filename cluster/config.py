"""Scheduler fields shared by model-specific campaign builders."""

SCHEDULER_KEYS = (
    "scheduler", "partition", "account", "qos", "slurm_user",
    "sbatch_command", "squeue_command", "pbs_user", "qsub_command", "qstat_command",
    "theory_partition", "fit_partition", "theory_account", "fit_account",
    "theory_qos", "fit_qos", "theory_walltime", "fit_walltime",
    "class_source_dir", "class_source_commit", "class_wrapper_sha256",
)


def scheduler_config(cluster):
    scheduler = str(cluster.get("scheduler", "pbs")).lower()
    if scheduler not in {"pbs", "slurm"}:
        raise ValueError(f"Unsupported cluster.scheduler: {scheduler!r}")
    return {key: cluster[key] for key in SCHEDULER_KEYS if key in cluster}
