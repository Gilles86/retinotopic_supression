#!/bin/bash -l
#SBATCH --job-name=loc_ts
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-651,u24-chaihm0-636
#SBATCH --cpus-per-task=1
#SBATCH --mem=26G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null

# Location time series and trials for the deconvolution (export_location_timeseries.py), one subject per task.
# Resources from fig3_event_average (1 CPU, 17-21 GB, 5.5 min per subject).
#   cd ~/retsupp_plan && sbatch --array=1-30 retsupp/visualize/paper/slurm_jobs/export_location_timeseries.sh
# Env: CODE_DIR (repo clone, default ~/retsupp_plan), EXTRA_ARGS.

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"

mkdir -p "$HOME/logs"
exec >"$HOME/logs/loc_ts_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"

export TMPDIR="/scratch/$USER/loc_ts_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.export_location_timeseries "$subject" ${EXTRA_ARGS:-}
echo "Finished $(date)"
