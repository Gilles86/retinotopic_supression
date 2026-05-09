#!/bin/bash
#SBATCH --account=zne.uzh
#SBATCH --job-name=retsupp_glmsingle
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#
# Fit GLMsingle single-trial betas for the retsupp visual-search task.
# Both sessions of a subject are fitted jointly (recommended: GLMsingle
# uses the session indicator to handle session-to-session scaling).
#
# Memory + time history:
#   96 GB / 6 h  -> OOM at TYPE-A ONOFF
#   128 GB / 6 h -> OOM (sub-08 MaxRSS=134G) and TIMEOUT (sub-18/19/21
#                   killed mid TYPE-D GLMDENOISE_RR at 6 h, MaxRSS up
#                   to 134 G)
#   192 GB / 12 h -> headroom for the largest subjects.
#
# Usage (single subject):
#   sbatch --export=PARTICIPANT_LABEL=05 fit_glmsingle.sh
#
# Usage (array job over all subjects):
#   sbatch --array=1-30 fit_glmsingle.sh
#
# Optional --export=KEY=VAL overrides:
#   SESSION         space-separated session numbers (default: both)
#   BOLD_TYPE       fmriprep | cleaned (default: fmriprep)
#   SMOOTHED        "1" to smooth BOLD before fitting (default: off)
#   DEBUG           "1" to write all 4 model steps + figures (default: off)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
fi

SESSION="${SESSION:-}"
BOLD_TYPE="${BOLD_TYPE:-fmriprep}"
SMOOTHED="${SMOOTHED:-0}"
DEBUG="${DEBUG:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-retsupp
REPO=$HOME/git/retsupp

LOGFILE="$HOME/logs/retsupp_glmsingle_sub-${PARTICIPANT_LABEL}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --bold-type "$BOLD_TYPE"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$SMOOTHED" = "1" ] && ARGS+=(--smoothed)
[ "$DEBUG"    = "1" ] && ARGS+=(--debug)

echo "fit_glmsingle: sub-${PARTICIPANT_LABEL}  bold=${BOLD_TYPE}  smoothed=${SMOOTHED}  debug=${DEBUG}"
echo "Args: ${ARGS[*]}"
echo "Started at $(date)"

# Conda activation (cluster: ~/data/miniforge3 + envs at ~/data/conda/envs/).
# Use the direct env binary to avoid pipe-buffering with conda run.
export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/retsupp/bin/python

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Build the retsupp env on the cluster:"
    echo "  see create_env/environment_cpu.yml in the repo."
    exit 1
fi

"$PYTHON" -u "$REPO/retsupp/glm/fit_glmsingle.py" "${ARGS[@]}"

echo "Finished at $(date)"
