#!/bin/bash -l
#SBATCH --job-name=bar_items
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-651,u24-chaihm0-636
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null

# Joint model of bar and items at an item location, one subject per task: export (raw BOLD, bar prediction, HP flag;
# ~5 min, 10 GB measured) then nideconv (deconvolve_bar_items.py). nideconv: ~/pylibs/nideconv (pip --target).
#   cd ~/retsupp_plan && sbatch --array=1-30 retsupp/visualize/paper/slurm_jobs/bar_items_joint.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11
mkdir -p "$HOME/logs" "$D/bar_items_joint"
exec >"$HOME/logs/bar_items_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname) sub-${subject} $(date)"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
PY="$HOME/data/conda/envs/retsupp_cuda/bin/python"
PYTHONPATH="$CODE_DIR" "$PY" -m retsupp.visualize.paper.export_location_timeseries "$subject" --subdir location_timeseries_joint
PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv" "$PY" -m retsupp.visualize.paper.deconvolve_bar_items \
    --folder "$D/location_timeseries_joint" --subject "$subject" --out "$D/bar_items_joint/sub-$(printf %02d $subject)_2026-11.tsv"
echo "Finished $(date)"
