#!/bin/bash
#SBATCH --job-name=prf_fit_run
#SBATCH --output=/home/gdehol/logs/retsupp_prf_run_%j.out  # Default SLURM log
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --mem=64G
# Usage: sbatch fit_run.sh <subject> <session> <run> <model>

# Load environment
. $HOME/init_conda.sh
module load gpu
source activate neural_priors2

subject=$1
session=$2
run=$3
model=$4

mkdir -p $HOME/logs

echo "Running fit_run.py for subject $subject, session $session, run $run, model $model"
python $HOME/git/retsupp/retsupp/modeling/fit_run.py $subject $session $run --model "$model" --r2_thr 0.04 \
	> $HOME/logs/fit_run_sub-${subject}_ses-${session}_run-${run}_model-${model}.out 2>&1