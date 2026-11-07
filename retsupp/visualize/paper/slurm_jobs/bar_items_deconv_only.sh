#!/bin/bash -l
#SBATCH --job-name=bar_items_deconv
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null

# Joint bar + items deconvolution only (export already in location_timeseries_joint), one subject per task.
#   cd ~/retsupp_plan && sbatch --array=1-30 retsupp/visualize/paper/slurm_jobs/bar_items_deconv_only.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11
mkdir -p "$HOME/logs" "$D/bar_items_joint"
exec >"$HOME/logs/bar_items_deconv_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
cd "$CODE_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.deconvolve_bar_items \
    --folder "$D/location_timeseries_joint" --subject "$subject" --out "$D/bar_items_joint/sub-$(printf %02d $subject)_2026-11.tsv"
echo "Finished $(date)"
