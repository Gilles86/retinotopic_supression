#!/bin/bash -l
#SBATCH --job-name=terms_pred
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Noise-free predictions per HP − LP term (predict_location_timeseries_terms.py: nohp, S, SD, SDT, SDT0) through the
# Fig. 3 voxel selection, then the relative deconvolution of each, one subject per task.
#   cd ~/retsupp_plan && sbatch --array=1-30 --exclude=u24-chaihm0-633 retsupp/visualize/paper/slurm_jobs/predict_location_timeseries_terms.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives
P=$D/paper_2026-11
SUB=location_timeseries_terms_pred_2026-11-11
mkdir -p "$HOME/logs"
exec >"$HOME/logs/terms_pred_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv" PYTHONUNBUFFERED=1   # nideconv is not in retsupp_cuda
PY="$HOME/data/conda/envs/retsupp_cuda/bin/python"
"$PY" -m retsupp.visualize.paper.predict_location_timeseries_terms "$subject" \
    --ladder-root "$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf" \
    --hrf-table "$P/roi_hrf_dog_final2.tsv" --trials-from "$P/location_timeseries_final2" --subdir "$SUB"
for v in nohp S SD SDT SDT0; do
    mkdir -p "$P/event_deconv_terms_pred_2026-11-11_$v"
    "$PY" -m retsupp.visualize.paper.deconvolve_location_responses --relative --subject "$subject" \
        --folder "$P/${SUB}_$v" --out "$P/event_deconv_terms_pred_2026-11-11_$v/sub-$(printf %02d "$subject")_2026-11.tsv"
done
echo "Finished $(date)"
