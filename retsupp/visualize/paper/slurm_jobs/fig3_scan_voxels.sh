#!/bin/bash -l
#SBATCH --job-name=fig3_scan
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --exclude=u24-chaihm0-651,u24-chaihm0-636
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/dev/null

# Figure 3c voxel/window scan, one (subject, ROI) per task: 9 subjects x 4 ROIs = 36.
#   cd ~/retsupp_plan && sbatch --array=1-36 retsupp/visualize/paper/slurm_jobs/fig3_scan_voxels.sh
# Env: CODE_DIR (repo clone, default ~/retsupp_plan), EXTRA_ARGS.

set -eo pipefail
SUBJECTS=(3 8 13 14 15 17 19 21 23)
ROIS=(V2 V3 hV4 V3AB)
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
subject="${SUBJECTS[$(( i / ${#ROIS[@]} ))]}"
roi="${ROIS[$(( i % ${#ROIS[@]} ))]}"
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"

mkdir -p "$HOME/logs"
exec >"$HOME/logs/fig3_scan_sub-${subject}_${roi}_${SLURM_JOB_ID}.txt" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" name="fig3_scan" 2>/dev/null || true
echo "Host $(hostname)  sub-${subject} ${roi}  started $(date)"

cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.fig3_scan_voxels "$subject" "$roi" ${EXTRA_ARGS:-}
echo "Finished $(date)"
