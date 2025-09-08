#!/bin/bash
#SBATCH --job-name=prf_conditionfit
#SBATCH --output=/home/gdehol/logs/retsupp_prf_conditionfit_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --mem=64G
# Usage: sbatch fit_condition.slurm <subject> <model>

# Load environment
. $HOME/init_conda.sh
module load gpu
source activate neural_priors2

subject=$1
model=$2

mkdir -p $HOME/logs

echo "Running fit_condition.py for subject $subject, model $model"
python $HOME/git/retsupp/retsupp/modeling/fit_condition.py $subject --model "$model" --bids_folder /shares/zne.uzh/gdehol/ds-retsupp --max_n_iterations 4000 --r2_thr 0.04 \
	> $HOME/logs/fit_condition_sub-${subject}_model-${model}.out 2>&1