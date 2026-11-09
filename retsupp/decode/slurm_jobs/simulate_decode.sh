#!/bin/bash -l
#SBATCH --job-name=decode_sim
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-651,u24-chaihm0-636
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Decoder simulation (simulate_decode.py): one task = one subject × run; per ROI the null (A = 0) and the HP gain
# field, free and outline decoders. Task index -> (subject, session, run) as in decode_final.sh.
#   cd ~/retsupp_plan && sbatch --array=1-360 retsupp/decode/slurm_jobs/simulate_decode.sh
#   PRF_SOURCE=jointhrf sbatch ...   (default canonical);  EXTRA_ARGS="--rois V1 V2 V3 --amplitudes 0 -0.075"

set -eo pipefail
idx0=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array} - 1 ))
subject=$(( idx0 / 12 + 1 ))
k=$(( idx0 % 12 ))
session=$(( k / 6 + 1 ))
run=$(( k % 6 + 1 ))
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/decode_sim_${PRF_SOURCE:-canonical}_sub-${subject}_ses-${session}_run-${run}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject} ses-${session} run-${run}  $(date)"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}" TF_NUM_INTEROP_THREADS=2
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.decode.simulate_decode "$subject" --session "$session" --run "$run" \
    --prf-source "${PRF_SOURCE:-canonical}" ${EXTRA_ARGS:-}
echo "Finished $(date)"
