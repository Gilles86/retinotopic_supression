#!/bin/bash
#SBATCH --job-name=prf_fit
#SBATCH --output=/home/gdehol/logs/retsupp_prf_%j.out  # Default SLURM log
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --constraint=A100
#SBATCH --mem=64G

# Load environment
. $HOME/init_conda.sh
module load gpu
source activate neural_priors2

# Define the fixed bids folder
bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

# Run the Python script with the provided arguments
python $HOME/git/retsupp/retsupp/modeling/fit_prf.py "$1" --model "$2" --bids_folder "$bids_folder" --r2_thr 0.0