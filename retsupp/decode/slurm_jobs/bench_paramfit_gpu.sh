#!/bin/bash
# GPU benchmark of ParameterFitter: patched (current submodule HEAD)
# vs original (the parent commit's optimize.py). Runs both versions
# back-to-back on the same GPU node to give a clean A/B.
#
#   sbatch retsupp/decode/slurm_jobs/bench_paramfit_gpu.sh
#
#SBATCH --job-name=bench_paramfit_gpu
#SBATCH --output=/dev/null
#SBATCH --time=20:00
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4|V100|A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -eo pipefail

LOGFILE="$HOME/logs/bench_paramfit_gpu_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "[$(date)] === GPU bench: ParameterFitter, original vs patched ==="

# u24 GPU CUDA setup
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
nvidia-smi --query-gpu=name --format=csv,noheader || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate retsupp_cuda
set -u
export PYTHONUNBUFFERED=1
# Don't disable GPU.
unset CUDA_VISIBLE_DEVICES

cd "$HOME/git/retsupp"

# Workload shapes to bench (V, R, T).
SHAPES=("200 25 100" "1000 30 258" "2000 40 258")

run_bench() {
    local label="$1"
    echo
    echo "=================================================================="
    echo "=== ${label}"
    echo "=================================================================="
    cd ~/git/retsupp/libs/braincoder
    git log -1 --format='%h %s' braincoder/optimize.py
    cd ~/git/retsupp
    for shape in "${SHAPES[@]}"; do
        IFS=' ' read -r V R T <<<"$shape"
        echo
        echo "--- shape V=${V} R=${R} T=${T} ---"
        # bench_paramfit.py disables GPU by default; strip that line.
        python -u -c "
import os, sys, time
import numpy as np, pandas as pd
import tensorflow as tf
print('TF:', tf.__version__, ' GPUs:', tf.config.list_logical_devices('GPU'))
from braincoder.hrf import SPMHRFModel
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.optimize import ParameterFitter

V, R, T = ${V}, ${R}, ${T}
grid_1d = np.linspace(-5, 5, R)
gx, gy = np.meshgrid(grid_1d, grid_1d)
grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
rng = np.random.default_rng(0)
pars = pd.DataFrame({
    'x': rng.uniform(-3, 3, V).astype(np.float32),
    'y': rng.uniform(-3, 3, V).astype(np.float32),
    'sd': rng.uniform(0.5, 2.0, V).astype(np.float32),
    'amplitude': rng.uniform(0.5, 2.0, V).astype(np.float32),
    'baseline': np.zeros(V, dtype=np.float32),
    'hrf_delay': np.full(V, 4.5, dtype=np.float32),
    'hrf_dispersion': np.full(V, 0.75, dtype=np.float32),
})
paradigm = (rng.random((T, R*R)) < 0.1).astype(np.float32)
hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
model = GaussianPRF2DWithHRF(
    grid_coordinates=grid, paradigm=paradigm, hrf_model=hrf,
    parameters=pars.astype(np.float32), flexible_hrf_parameters=True)
pred = model.predict()
bold = pred.values + rng.normal(0, 0.05, pred.shape).astype(np.float32)
bold_df = pd.DataFrame(bold, columns=pred.columns)
pf = ParameterFitter(model=model, data=bold_df, paradigm=paradigm)
init = pars.copy()
init['x'] = init['x'] + rng.normal(0, 0.3, V).astype(np.float32)
init['y'] = init['y'] + rng.normal(0, 0.3, V).astype(np.float32)
# Warm up.
pf.fit(init_pars=init, learning_rate=0.01,
       max_n_iterations=2, min_n_iterations=10_000, progressbar=False)
N = 50
t0 = time.time()
pf.fit(init_pars=init, learning_rate=0.01,
       max_n_iterations=N, min_n_iterations=10_000, progressbar=False)
ms = (time.time() - t0) * 1000 / N
print(f'V={V} R={R} T={T}: {ms:.2f} ms/iter')
"
    done
}

# 1) PATCHED — current submodule HEAD (= 4acbd4b)
run_bench "PATCHED (current HEAD: ParameterFitter.fit @tf.function)"

# 2) Revert optimize.py to parent commit and re-bench. Use git checkout
#    so we only touch the one file; restore at the end.
cd ~/git/retsupp/libs/braincoder
git checkout 25fd483 -- braincoder/optimize.py
cd ~/git/retsupp

run_bench "ORIGINAL (parent commit 25fd483: ParameterFitter eager)"

# Restore.
cd ~/git/retsupp/libs/braincoder
git checkout 4acbd4b -- braincoder/optimize.py
cd ~/git/retsupp

echo
echo "[$(date)] done"
