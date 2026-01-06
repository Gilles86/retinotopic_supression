#!/bin/bash
#SBATCH --job-name=create_retsupp_cpu_env
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/create_cpu_env_%j.log

set -e

echo "=== Creating CPU-only conda environment ==="
echo "Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f retsupp-environment.yml --yes

echo ""
echo "=== Environment created successfully ==="
echo "Finished at $(date)"
echo ""
echo "To activate the environment, run:"
echo "  conda activate retsupp_cpu"
