#!/bin/bash -l
#SBATCH --job-name=decode_relnull
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

# Voxel-selection robustness of Fig. 6: decode_final.py with --selection relnull (fold-wise cross-run reliability
# r > MIN_R and circular-shift p <= .01 on the fold's training runs, SD <= 20%, cap 2000), jointhrf pRFs, all nine areas.
# Needs derivatives/prf_cv_roi_reliability_null_folds (voxel_reliability_null.py --folds).
# One task = one subject × run, subjects 3-30: idx0 = task - 1; subject = idx0 / 12 + 3; k = idx0 % 12;
# session = k / 6 + 1; run = k % 6 + 1 (subjects with 5 runs in a session exit cleanly for run 6).
#   cd ~/retsupp_plan && MIN_R=0.2 sbatch --array=1-336 retsupp/decode/slurm_jobs/decode_final_relnull.sh
# Outputs: derivatives/decoded_final_2026-11/jointhrf_relnull_r<MIN_R>/<decoder>/sub-XX/

set -eo pipefail
idx0=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array} - 1 ))
subject=$(( idx0 / 12 + 3 ))
k=$(( idx0 % 12 ))
session=$(( k / 6 + 1 ))
run=$(( k % 6 + 1 ))
MIN_R="${MIN_R:?set MIN_R}"
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/decode_relnull_r${MIN_R}_sub-${subject}_ses-${session}_run-${run}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject} ses-${session} run-${run}  r > ${MIN_R}  $(date)"
export TMPDIR=/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}" TF_NUM_INTEROP_THREADS=2
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.decode.decode_final "$subject" --session "$session" --run "$run" \
    --prf-source jointhrf --selection relnull --min-r "$MIN_R" \
    --rois V1 V2 V3 V3AB IPS hV4 VO LO TO --decoders ${DECODERS:-free outline} ${EXTRA_ARGS:-}
echo "Finished $(date)"
