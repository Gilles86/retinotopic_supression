#!/bin/bash
#SBATCH --job-name=backup_gecon
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
# Add-only backup of ds-retsupp modelling inputs to the g_econ share (retsupp/prepare/backup_to_gecon.py).
# One task per unit (python -m retsupp.prepare.backup_to_gecon --list-units). VERIFY=1 checksums instead of copying.
#   kinit -c /scratch/gdehol/krb5cc gdehol@D.UZH.CH      (the user, beforehand)
#   cd ~/retsupp_plan && sbatch --array=1-63%1 retsupp/prepare/slurm_jobs_backup_to_gecon.sh
set -eo pipefail
mkdir -p "$HOME/logs"
exec >"$HOME/logs/backup_gecon_${VERIFY:+verify_}${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt" 2>&1
export KRB5CCNAME=/scratch/gdehol/krb5cc
export TMPDIR="/scratch/gdehol/tmp/gecon_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/retsupp_plan"
exec python3 -u -m retsupp.prepare.backup_to_gecon --index "$SLURM_ARRAY_TASK_ID" ${VERIFY:+--verify}
