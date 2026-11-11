#!/bin/bash -l
#SBATCH --job-name=decode_model_pred
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null

# Decode the joint gain-field fit's noise-free prediction (and the real data on the same voxels) with the Fig. 6 decoder
# (retsupp/decode/decode_model_prediction.py). One task = one subject (array index = subject).
#   cd ~/retsupp_plan && sbatch --array=3-30 retsupp/decode/slurm_jobs/decode_model_prediction.sh
#   EXTRA_ARGS="--runs 1:1 --out-root /scratch/gdehol/decode_model_canary" sbatch --array=23 ...   (canary)

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives
mkdir -p "$HOME/logs"
exec >"$HOME/logs/decode_model_pred_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}" TF_NUM_INTEROP_THREADS=2
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.decode.decode_model_prediction "$subject" \
    --ladder-root "$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf" \
    --hrf-table "$D/paper_2026-11/roi_hrf_dog_final2.tsv" ${EXTRA_ARGS:-}
echo "Finished $(date)"
