#!/bin/bash
#SBATCH --job-name=create_retsupp_gpu_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/create_gpu_env_%j.log 2>&1

set -e

echo "=== Creating CUDA-enabled conda environment ==="
echo "Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU info:"
nvidia-smi

# Load CUDA 12.1 module
module load cuda/12.6.3

# Activate conda (adjust path if needed)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create environment
conda env create -f environment_cuda.yml --yes

echo ""
echo "=== Environment created successfully ==="
echo "Finished at $(date)"
echo ""
echo "To activate the environment, run:"
echo "  conda activate retsupp_cuda"
