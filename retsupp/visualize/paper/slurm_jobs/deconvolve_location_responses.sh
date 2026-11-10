#!/bin/bash -l
#SBATCH --job-name=deconv
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

# nideconv deconvolution per subject (deconvolve_location_responses.py) on the exported location time series.
# nideconv is not in retsupp_cuda; it is installed with --target in ~/pylibs/nideconv.
#   cd ~/retsupp_plan && sbatch --array=1-30 retsupp/visualize/paper/slurm_jobs/deconvolve_location_responses.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11
IN="${IN_SUBDIR:-location_timeseries}"; OUTD="${OUT_SUBDIR:-event_deconv}"
mkdir -p "$HOME/logs" "$D/$OUTD"
exec >"$HOME/logs/deconv_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.deconvolve_location_responses \
    --folder "$D/$IN" --subject "$subject" --out "$D/$OUTD/sub-$(printf %02d $subject)_2026-11.tsv" ${EXTRA_ARGS:-}
echo "Finished $(date)"
