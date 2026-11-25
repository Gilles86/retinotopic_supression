#!/bin/bash -l
#SBATCH --job-name=deconv_jk
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# Fig. 3 deconvolution with leave-one-run-out and odd/even fits (deconvolve_location_responses.py --jackknife), plain
# (item x HP/LP) and --relative (item x HP / next / opposite), on the final location time series, one subject per task.
#   cd <code dir> && CODE_DIR=$PWD sbatch --array=3-30 retsupp/visualize/paper/slurm_jobs/deconv_jackknife.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
P=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11
TAG="${TAG:-2026-11-25}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/deconv_jk_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
for mode in plain relative; do
    out="$P/event_deconv_final2_jk_${mode}_${TAG}"
    mkdir -p "$out"
    extra=""; [ "$mode" = relative ] && extra="--relative"
    "$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.deconvolve_location_responses \
        --jackknife $extra --subject "$subject" --folder "$P/location_timeseries_final2" \
        --out "$out/sub-$(printf %02d "$subject")_2026-11.tsv"
done
echo "Finished $(date)"
