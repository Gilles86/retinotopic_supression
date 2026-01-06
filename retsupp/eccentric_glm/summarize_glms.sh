#!/bin/bash
#SBATCH --job-name=summarize_eccentric_glm
#SBATCH --output=/home/gdehol/logs/retsupp_summarize_eccentric_glm_%j_%a.out
#SBATCH --time=25:00
#SBATCH -c 16
#SBATCH --mem=32G

# Load environment
. $HOME/init_conda.sh
source activate retsupp

export subject=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

echo "Fitting glm"
python $HOME/git/retsupp/retsupp/eccentric_glm/summarize_glms.py $subject --bids_folder /shares/zne.uzh/gdehol/ds-retsupp