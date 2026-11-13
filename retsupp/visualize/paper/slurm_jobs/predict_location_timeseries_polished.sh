#!/bin/bash -l
#SBATCH --job-name=polished_pred
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null

# Noise-free predictions of the refitted joint fit per HP term (predict_location_timeseries_polished.py: nohp, S, SD,
# SDT, SDT0) through the Fig. 3 voxel selection of the refit, then both deconvolutions (relative and HP/LP) of each,
# plus the HP/LP deconvolution of the measured refit series. One subject per task.
#   cd ~/retsupp_plan && sbatch --array=3-30 retsupp/visualize/paper/slurm_jobs/predict_location_timeseries_polished.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives
P=$D/paper_2026-11
SUB=location_timeseries_polished_pred_2026-11-13
DEC=event_deconv_polished_pred_2026-11-13
MEAS=location_timeseries_polished_2026-11-12
S2=$(printf %02d "$subject")
mkdir -p "$HOME/logs"
exec >"$HOME/logs/polished_pred_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv" PYTHONUNBUFFERED=1   # nideconv is not in retsupp_cuda
PY="$HOME/data/conda/envs/retsupp_cuda/bin/python"
"$PY" -m retsupp.visualize.paper.predict_location_timeseries_polished "$subject" \
    --root "$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished" \
    --dt-root "$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf" \
    --hrf-table "$P/roi_hrf_dog_final2.tsv" --trials-from "$P/$MEAS" --subdir "$SUB"
for v in nohp S SD SDT SDT0; do
    for mode in relative hplp; do
        o="$P/${DEC}_${mode}_$v"; mkdir -p "$o"
        "$PY" -m retsupp.visualize.paper.deconvolve_location_responses $([ $mode = relative ] && echo --relative) \
            --subject "$subject" --folder "$P/${SUB}_$v" --out "$o/sub-${S2}_2026-11.tsv"
    done
done
o="$P/event_deconv_hplp_polished_2026-11-13"; mkdir -p "$o"
"$PY" -m retsupp.visualize.paper.deconvolve_location_responses --subject "$subject" --folder "$P/$MEAS" \
    --out "$o/sub-${S2}_2026-11.tsv"
echo "Finished $(date)"
