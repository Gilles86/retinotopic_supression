#!/bin/bash
#SBATCH --job-name=dog_dyn_target_shS
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:25:00

# v3 + target ('phasic capture') joint AF + DoG-PRF braincoder fit,
# with sigma_T_dyn TIED to sigma_dyn (shared phasic σ).
#
# Tests whether the target-onset Gaussian's spatial extent really
# differs from the distractor-onset one, or whether the larger
# σ_T_dyn estimates seen in V3AB / VO are identifiability slack.
#
# Same 8 shared parameters as the un-tied v3+target fit, but the
# σ_T_dyn raw variable is functionally inert: every forward pass
# overrides post-softplus σ_T_dyn := σ_dyn before the loss.
#
# Output:
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma/sub-XX/...
#
# Submission: 30 subjects x 8 ROIs = 240 array tasks.
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_sharedSigma.sh
#
# The array task id maps to (subject_idx, roi):
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // 8 ]
#   roi     = ROIS[    idx0 %  8 ]

set -eo pipefail   # no -u: conda's activate-*.d/ touches unbound vars

sleep $(( RANDOM % 30 ))   # NFS dogpile defense; harmless when sparse

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_v3_target_sharedSigma_${SLURM_JOB_NAME:-target_shS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        v3 + target + sharedSigma (sigma_T_dyn := sigma_dyn)"

# --- Subject + ROI decoding from SLURM_ARRAY_TASK_ID. ---
SUB_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-330." >&2
    exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_ROIS ))
roi_idx=$(( idx0 % N_ROIS ))
if [[ "$sub_idx" -ge "$N_SUBS" ]]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range; max is $(( N_SUBS * N_ROIS ))." >&2
    exit 2
fi
subject="${SUB_IDS[$sub_idx]}"
roi="${ROIS[$roi_idx]}"

echo "Subject:     ${subject}"
echo "ROI:         ${roi}"

# --- Conda env (CUDA env, forced to CPU per fit_dog_dynamic_af_braincoder_cpu.sh). ---
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dynamic_af_braincoder.py"

echo "Running v3 + target + sharedSigma fit for sub-${subject}, roi=${roi}"

# Voxel selection: posterior-based via the logit-GMM p_signal NIfTI.
#   PSIGNAL_THR (default 0.95)        → keep voxels with P(signal | R²) > this
#   APERTURE_MASS_THR (default 0)     → optional: also require PRF mass inside
#                                       bar aperture ≥ this. 0 to skip.
# Variants are segregated by output_subdir (Python appends _pSig{thr}[_apt{thr}]).

# Model-family knobs (default: base sharedSigma — σ_T_dyn := σ_dyn only):
#   SHARED_DYN_GAIN=1   → additionally tie g_LP_dyn := g_HP_dyn
#   ALL_SHARED_SIGMA=1  → additionally tie σ_AF := σ_dyn (single AF width)
# Either or both can be enabled; output_subdir is suffixed accordingly.
MODEL_FLAGS=""
[[ "${SHARED_DYN_GAIN:-0}" == "1" ]] && MODEL_FLAGS="$MODEL_FLAGS --shared-dyn-gain"
[[ "${ALL_SHARED_SIGMA:-0}" == "1" ]] && MODEL_FLAGS="$MODEL_FLAGS --all-shared-sigma"

# Base PRF model. Default 4 (DoG+HRF, canonical). Set MODEL_LABEL=6
# (DN+HRF) to fit AF on a different base; output_subdir auto-suffixes
# with `_base-mN` so variants don't clobber each other.
MODEL_LABEL="${MODEL_LABEL:-4}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --model-label "$MODEL_LABEL" \
    --with-target \
    --shared-target-sigma \
    $MODEL_FLAGS \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --sigma-t-dyn-init 2.0 \
    --g-t-dyn-init 0.0 \
    --p-signal-thr "${PSIGNAL_THR:-0.5}" \
    --aperture-mass-thr "${APERTURE_MASS_THR:-0}"

echo "Finished:    $(date)"
