#!/bin/bash
# SLURM array: one (l2, lr) cell per task.
#
# Cell grid (12 cells): l2 in {0.01, 0.05, 0.1, 0.5} x lr in {0.01, 0.05, 0.1}.
# Index map: idx = l2_idx * 3 + lr_idx + 1  (1..12).
#
# Submit:
#   sbatch --array=1-12 retsupp/decode/slurm_jobs/smoke_test_sweep_array.sh           # defaults: sub-02 V1 ses-1 run-1
#   sbatch --array=1-12 retsupp/decode/slurm_jobs/smoke_test_sweep_array.sh 5 V1 1 1  # sub-05 V1 ses-1 run-1
#
# After all tasks finish, run the aggregator (separate job, see
# smoke_test_sweep_aggregate.sh) OR locally via:
#   ~/mambaforge/envs/retsupp/bin/python -m retsupp.decode.smoke_test_sweep \
#       --subject 2 --roi V1 --session 1 --run 1 --aggregate-only
#SBATCH --job-name=decode_sweep_cell
#SBATCH --output=/dev/null
#SBATCH --time=30:00
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

SUBJECT="${1:-2}"
ROI="${2:-V1}"
SESSION="${3:-1}"
RUN="${4:-1}"

LOGFILE="$HOME/logs/decode_sweep_cell_sub-${SUBJECT}_${ROI}_ses-${SESSION}_run-${RUN}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

# Spread out NFS profile reads.
sleep $(( (RANDOM % 15) + 1 ))

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

L2S=(0.01 0.05 0.1 0.5)
LRS=(0.01 0.05 0.1)

idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
l2_idx=$(( idx / ${#LRS[@]} ))
lr_idx=$(( idx % ${#LRS[@]} ))

if [ "$l2_idx" -ge "${#L2S[@]}" ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID out of range; exiting."
    exit 0
fi

L2=${L2S[$l2_idx]}
LR=${LRS[$lr_idx]}

scontrol update jobid="${SLURM_JOB_ID}" \
    name="decode_sweep_sub-$(printf %02d "$SUBJECT")_${ROI}_l2-${L2}_lr-${LR}" \
    2>/dev/null || true

echo "[$(date)] sub-${SUBJECT} ${ROI} ses-${SESSION} run-${RUN}  L2=${L2}  lr=${LR}  on $(hostname)"

python -u -m retsupp.decode.smoke_test_sweep \
    --bids-folder "$bids_folder" \
    --subject "$SUBJECT" \
    --roi "$ROI" \
    --session "$SESSION" \
    --run "$RUN" \
    --l2-norms "$L2" \
    --learning-rates "$LR"

echo "[$(date)] done"
