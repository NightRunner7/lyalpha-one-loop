#!/usr/bin/env bash
#PBS -N lya_lcdm_loop
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

# Edit only these three lines for the cluster installation.
PROJECT_DIR="/path/to/lyalpha_one_loop"
PYTHON_BIN="/path/to/class-enabled/python"
MODEL_PRESET="lcdm_2021_massless"

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" "$PYTHON_BIN" scripts/generate_theory.py \
  --preset "$MODEL_PRESET" \
  --quality precision \
  --output "results/${MODEL_PRESET}_precision.npz" \
  --checkpoint-dir "results/checkpoints_${MODEL_PRESET}"
