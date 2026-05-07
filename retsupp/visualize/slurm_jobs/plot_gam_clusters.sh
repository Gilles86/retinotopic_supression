#!/bin/bash
#SBATCH --job-name=gam_clusters
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Per-ROI hierarchical Bayesian GAM (bambi/HSGP) on conditionwise PRF
# projection-vs-distance data — cluster CPU run with aggressive sampling.
#
# Single job (no array): all ROIs run sequentially in the same job.
# bambi 0.17 + pymc 5.28 are available in the `bauer` env on the cluster.
#
# Usage:
#   sbatch retsupp/visualize/slurm_jobs/plot_gam_clusters.sh
#
# Outputs:
#   /shares/zne.uzh/gdehol/ds-retsupp/derivatives/gam_clusters/gam_clusters.pdf
#   /shares/zne.uzh/gdehol/ds-retsupp/derivatives/gam_clusters/<roi>.nc  (InferenceData)

set -euo pipefail

LOGFILE="$HOME/logs/gam_clusters_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate bauer

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# pytensor compile dir per-job to avoid lockfile contention.
export PYTENSOR_FLAGS="base_compiledir=$HOME/.pytensor_${SLURM_JOB_ID}"

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
out_dir="${bids_folder}/derivatives/gam_clusters"
mkdir -p "$out_dir"
out_pdf="${out_dir}/gam_clusters.pdf"

PYTHON="$HOME/data/conda/envs/bauer/bin/python"

# Sampling strategy:
# - nutpie backend: Rust-based NUTS, ~5-10x faster than PyMC's, uses a
#   full mass matrix by default. The right tool for HSGP-shaped posteriors
#   where basis-weight ↔ length-scale correlations make a diagonal mass
#   matrix inefficient (and bumping target_accept alone doesn't fix it).
# - draws=3000, tune=3000 (no need to crank target_accept hard with nutpie).
# - target_accept=0.95 — nutpie default; relax from the previous 0.995.
# - chains=4.
"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/visualize/plot_gam_clusters.py" \
    --bids-folder "$bids_folder" \
    --out "$out_pdf" \
    --nuts-sampler nutpie \
    --draws 3000 \
    --tune 3000 \
    --target-accept 0.95 \
    --chains 4 \
    --save-traces

echo "Finished: $(date)"
echo "PDF:      ${out_pdf}"
ls -lh "${out_dir}" | head -20
