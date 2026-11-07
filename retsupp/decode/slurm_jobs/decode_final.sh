#!/bin/bash -l
#SBATCH --job-name=decode_final
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-651,u24-chaihm0-636
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null

# Gain-free decoding for the paper (decode_final.py): one task = one subject × run, V1-V3 × both decoders.
# Task index -> (subject, session, run): idx0 = task - 1; subject = idx0 / 12 + 1; k = idx0 % 12;
# session = k / 6 + 1; run = k % 6 + 1 (subjects with 5 runs in a session exit cleanly for run 6).
#   cd ~/retsupp_plan && sbatch --array=1-360 retsupp/decode/slurm_jobs/decode_final.sh
#   PRF_SOURCE=jointhrf sbatch ...   (default canonical);  EXTRA_ARGS="--rois V1 --decoders outline"
# Resources: May decode.py tasks (one ROI, one decoder) took ~1.4 min on 8 CPUs, 16 GB.

set -eo pipefail
idx0=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array} - 1 ))
subject=$(( idx0 / 12 + 1 ))
k=$(( idx0 % 12 ))
session=$(( k / 6 + 1 ))
run=$(( k % 6 + 1 ))
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/decode_final_${PRF_SOURCE:-canonical}_sub-${subject}_ses-${session}_run-${run}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject} ses-${session} run-${run}  $(date)"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}" TF_NUM_INTEROP_THREADS=2
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.decode.decode_final "$subject" --session "$session" --run "$run" \
    --prf-source "${PRF_SOURCE:-canonical}" ${EXTRA_ARGS:-}
echo "Finished $(date)"
