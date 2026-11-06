#!/bin/bash -l
#SBATCH --job-name=fig3_evavg
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-651,u24-chaihm0-636
#SBATCH --cpus-per-task=2
#SBATCH --mem=26G
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null

# Event-locked average of data − m0 and m7 − m0 (fig3_event_average.py), one subject per task, nine ROIs.
# Resources from the voxel scan (fig3_scan_voxels: ~1 CPU, 20-22 GB, 3-4 min per ROI).
#   cd ~/retsupp_plan && sbatch --array=1-30 retsupp/visualize/paper/slurm_jobs/fig3_event_average.sh
# Env: CODE_DIR (repo clone, default ~/retsupp_plan), EXTRA_ARGS.

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"

mkdir -p "$HOME/logs"
exec >"$HOME/logs/fig3_evavg_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"

cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.fig3_event_average "$subject" ${EXTRA_ARGS:-}
echo "Finished $(date)"
