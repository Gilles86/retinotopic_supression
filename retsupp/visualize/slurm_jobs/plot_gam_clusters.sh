#!/bin/bash
#SBATCH --job-name=gam_clusters
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# Per-ROI hierarchical Bayesian GAM (bambi/HSGP) on conditionwise PRF
# projection-vs-distance data, parallelised as a SLURM array.
#
# 8 ROIs × 2 model variants = 16 array tasks:
#   form 'population' : proj ~ hsgp(distance) + (1|subject)
#   form 'subjectwise': adds hsgp(distance, by=subject, share_cov=True)
#
# Each task fits ONE (ROI, form) and saves:
#   .../derivatives/gam_clusters/<form>/<roi>.pdf
#   .../derivatives/gam_clusters/<form>/<roi>_*_idata.nc
# Local rendering can stitch all per-ROI PDFs together later.
#
# bauer env (bambi 0.17.2 + pymc 5.28 + nutpie + numpyro + jax).
#
# Submit:
#   sbatch --array=1-16 retsupp/visualize/slurm_jobs/plot_gam_clusters.sh

set -euo pipefail

LOGFILE="$HOME/logs/gam_clusters_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0}"
echo "Started: $(date)"

# --- (ROI, model_form) decoding from SLURM_ARRAY_TASK_ID. ---
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
FORMS=(population subjectwise)
N_ROIS=${#ROIS[@]}
N_FORMS=${#FORMS[@]}
N_TASKS=$(( N_ROIS * N_FORMS ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-${N_TASKS}." >&2
    exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
form_idx=$(( idx0 / N_ROIS ))
roi_idx=$(( idx0 % N_ROIS ))
roi="${ROIS[$roi_idx]}"
form="${FORMS[$form_idx]}"

echo "ROI:        ${roi}"
echo "Model form: ${form}"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate bauer

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTENSOR_FLAGS="base_compiledir=$HOME/.pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
out_dir="${bids_folder}/derivatives/gam_clusters/${form}"
mkdir -p "$out_dir"
out_pdf="${out_dir}/${roi}.pdf"

PYTHON="$HOME/data/conda/envs/bauer/bin/python"

# nutpie backend uses a full mass matrix → handles HSGP correlations
# without needing target_accept cranked all the way up.
"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/visualize/plot_gam_clusters.py" \
    --bids-folder "$bids_folder" \
    --rois "${roi}" \
    --out "$out_pdf" \
    --model-form "${form}" \
    --nuts-sampler nutpie \
    --draws 3000 \
    --tune 3000 \
    --target-accept 0.95 \
    --chains 4 \
    --save-traces

echo "Finished: $(date)"
echo "PDF:      ${out_pdf}"
ls -lh "${out_dir}/" | head -10
