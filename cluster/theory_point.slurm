#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -Eeuo pipefail
# Execute the same scientific worker as PBS. Its #PBS lines are comments
# when invoked by bash; Slurm resources are set above/by campaign_manager.
exec bash "${PROJECT_DIR:?PROJECT_DIR is required}/cluster/theory_point.pbs"
