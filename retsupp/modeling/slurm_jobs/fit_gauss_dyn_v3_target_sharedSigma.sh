#!/bin/bash
#SBATCH --job-name=gauss_dyn_target_shS
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:25:00

# v3 + target + sharedSigma joint AF + GAUSSIAN-PRF braincoder fit.
# Default init: model 3 (Gaussian + flex HRF) per the "HRF optim" canonical.
#
# Voxel selection: posterior-based via the logit-GMM p_signal NIfTI.
#   PSIGNAL_THR (default 0.5)         → keep voxels with P(signal|R²) > this
#   APERTURE_MASS_THR (default 0)     → optional aperture-mass filter
# Model variants (factorial with DoG):
#   SHARED_DYN_GAIN=1   → tie g_LP_dyn := g_HP_dyn
#   ALL_SHARED_SIGMA=1  → tie σ_AF := σ_dyn := σ_T_dyn
# MODEL (default 3) selects the mean-PRF NIfTI to init from.

set -eo pipefail

sleep $(( RANDOM % 30 ))

LOGFILE="$HOME/logs/gauss_dyn_v3_target_sharedSigma_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        v3 + target + sharedSigma + Gaussian-PRF init"

SUB_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_ROIS ))
roi_idx=$(( idx0 % N_ROIS ))
if [[ "$sub_idx" -ge "$N_SUBS" ]]; then
    echo "ERROR: array index out of range." >&2; exit 2
fi
subject="${SUB_IDS[$sub_idx]}"
roi="${ROIS[$roi_idx]}"

MODEL="${MODEL:-3}"   # 1 = Gaussian no-HRF, 3 = Gaussian + flex HRF (canonical)

scontrol update jobid="${SLURM_JOB_ID}" \
    name="gauss_af_m${MODEL}_sub-$(printf %02d $subject)_${roi}" 2>/dev/null || true

echo "Subject:     ${subject}"
echo "ROI:         ${roi}"
echo "PRF model:   ${MODEL}"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_gaussian_dynamic_af_braincoder.py"

R2_FLAG=""
[[ "${USE_FDR:-0}" == "1" ]] && R2_FLAG="--r2_thr -1"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --model-label "$MODEL" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --with-target \
    --shared-target-sigma \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --sigma-t-dyn-init 2.0 \
    --g-t-dyn-init 0.0 \
    $R2_FLAG

echo "Finished:    $(date)"
