#!/usr/bin/env bash
#PBS -N lya_custom_loop
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

# Edit these values for the cluster and for the selected model.
PROJECT_DIR="/path/to/lyalpha_one_loop"
PYTHON_BIN="/path/to/custom-class/python"
MODEL_JSON="examples/custom_lcdm_model.json"
RUN_NAME="my_model"
QUALITY="precision"

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" "$PYTHON_BIN" scripts/generate_theory.py \
  --model-json "$MODEL_JSON" \
  --quality "$QUALITY" \
  --output "results/${RUN_NAME}_${QUALITY}.npz" \
  --checkpoint-dir "results/checkpoints_${RUN_NAME}_${QUALITY}"
