#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=/home/gdehol/logs/retsupp_preprocess_%A_%a.out
#SBATCH --error=/home/gdehol/logs/retsupp_preprocess_%A_%a.out
#SBATCH --array=1-20  # Adjust the rande to your number of subjects
#SBATCH --time=20:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16

# Load environment
. $HOME/init_conda.sh
source activate neural_priors2

SUBJECT=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

echo "Processing subject: sub-${SUBJECT}"

for SES in 1 2; do
    echo "Running clean.py for sub-${SUBJECT} ses-${SES}"
    python $HOME/git/retsupp/retsupp/preprocess/clean.py --subject sub-${SUBJECT} --session ${SES}
done

echo "Running mean_ts.py for sub-${SUBJECT}"
python $HOME/git/retsupp/retsupp/preprocess/mean_ts.py --subject sub-${SUBJECT}

echo "Done for subject: sub-${SUBJECT}"
