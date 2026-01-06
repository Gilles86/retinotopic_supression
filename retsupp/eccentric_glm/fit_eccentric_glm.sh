#!/bin/bash
#SBATCH --job-name=prf_conditionfit
#SBATCH --output=/home/gdehol/logs/retsupp_glm_eccentric_%j_%a.out
#SBATCH --time=1:00:00
#SBATCH -c 16
#SBATCH --mem=64G

# Load environment
. $HOME/init_conda.sh
source activate retsupp

export subject=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

echo "Cleaning up data"
python $HOME/git/retsupp/retsupp/eccentric_glm/clean_data.py $subject --bids_folder /shares/zne.uzh/gdehol/ds-retsupp

echo "Fitting glm"
python $HOME/git/retsupp/retsupp/eccentric_glm/fit_glm.py $subject --bids_folder /shares/zne.uzh/gdehol/ds-retsupp