#!/usr/bin/env bash
set -Eeuo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${SUITE_DIR:?SUITE_DIR is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -u -m campaigns.accdm.precision_suite \
    --suite "$SUITE_DIR" run-point --task-id "$SLURM_ARRAY_TASK_ID"
