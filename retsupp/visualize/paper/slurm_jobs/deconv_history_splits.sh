#!/bin/bash -l
#SBATCH --job-name=deconv_hist
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

# Richter follow-up A1/A2: the Fig. 3 relative deconvolution (deconvolve_location_responses.py --relative) on the final
# location time series, separately per session, per block within session (the session's first or second HP location)
# and per session x block, one subject per task (sub-03-30: the regular HP schedule).
#   cd ~/retsupp_plan && sbatch --array=3-30 --exclude=u24-chaihm0-633 retsupp/visualize/paper/slurm_jobs/deconv_history_splits.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
P=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11
mkdir -p "$HOME/logs"
exec >"$HOME/logs/deconv_hist_sub-${subject}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject}  started $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HOME/pylibs/nideconv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"   # nideconv is not in retsupp_cuda
for split in ses1 ses2 block1 block2 s1b1 s1b2 s2b1 s2b2; do
    out="$P/event_deconv_relative_final2_split_2026-11-11_$split"
    mkdir -p "$out"
    "$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.visualize.paper.deconvolve_location_responses \
        --relative --split "$split" --subject "$subject" --folder "$P/location_timeseries_final2" \
        --out "$out/sub-$(printf %02d "$subject")_2026-11.tsv"
done
echo "Finished $(date)"
