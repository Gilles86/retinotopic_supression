#!/bin/bash -l
#SBATCH --job-name=kappa_pred
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/dev/null

# Noise-free predictions of m7_full, m8_full_dogk and dogk with kappa = 0 through the Fig. 3 voxel selection
# (predict_location_timeseries_kappa.py), one subject per task.
#   cd ~/retsupp_plan && sbatch --array=1-30 retsupp/visualize/paper/slurm_jobs/predict_location_timeseries_kappa.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives
mkdir -p "$HOME/logs"
exec >"$HOME/logs/kappa_pred_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.predict_location_timeseries_kappa "$subject" \
    --ladder-root "$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf" \
    --dogk-root "$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_dogcs_2026-11" \
    --hrf-table "$D/paper_2026-11/roi_hrf_dog_final2.tsv" \
    --trials-from "$D/paper_2026-11/location_timeseries_final2"
echo "Finished $(date)"
