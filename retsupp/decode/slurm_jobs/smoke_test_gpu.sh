#!/bin/bash
# GPU smoke test: same sub-02 V1 ses-1 run-1 config as smoke_test.sh
# but on a GPU node. Use to compare wall-clock vs the CPU smoke.
#   sbatch retsupp/decode/slurm_jobs/smoke_test_gpu.sh
#SBATCH --job-name=decode_smoke_gpu
#SBATCH --output=/dev/null
#SBATCH --time=20:00
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4|V100|A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-decode_smoke_gpu}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

module load gpu

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
# retsupp_cuda has braincoder + TF-CUDA. The cluster `retsupp` env is
# incomplete (see CLAUDE.md §"Cluster env quirks").
set +u
conda activate retsupp_cuda
set -u

export PYTHONUNBUFFERED=1
# Tag the GPU output paths so they don't clobber the CPU smoke outputs.
TAG="gpu"

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

cd "$HOME/git/retsupp"

echo "[$(date)] === GPU smoke test ==="
echo "node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

START=$(date +%s)
python -u -m retsupp.decode.smoke_test \
    --bids-folder "$bids_folder" \
    --subject 2 --roi V1 \
    --session 1 --run 1 \
    --resolution 30 --max-voxels 200 \
    --max-n-iterations 600 --resid-max-iter 300 \
    --out-npz "notes/data/decoded_smoke_sub-02_V1_${TAG}.npz" \
    --out-fig "notes/figures/decoded_smoke_sub-02_V1_${TAG}.pdf" \
    --out-tsv "notes/data/decoded_smoke_sub-02_V1_${TAG}_ring.tsv"
END=$(date +%s)
echo "[$(date)] done (wall-clock ${TAG}: $((END - START)) s)"
