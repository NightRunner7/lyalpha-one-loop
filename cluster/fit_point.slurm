#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

set -Eeuo pipefail
exec bash "${PROJECT_DIR:?PROJECT_DIR is required}/cluster/fit_point.pbs"
