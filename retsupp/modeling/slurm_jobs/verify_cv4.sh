#!/bin/bash
#SBATCH --job-name=cv4_verify
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --constraint="L4|V100|A100"
#SBATCH --mem=16G

# Small-scale smoke test of the 4-level CV pipeline (null/model0/gain/shift)
# for ONE (subject, ROI). Confirms:
#   1. model0, gain, shift each produce finite per-voxel CV-R²,
#   2. voxel_ids are IDENTICAL across model0/gain/shift TSVs,
#   3. the three cv-r2.tsv / cv-params.tsv / meta.json files are well-formed,
#   4. the aggregator runs, computes the model-free null level, asserts
#      voxel-id identity, and emits the per-ROI + threshold-sweep tables.
#
# Defaults: sub-3, V3AB, max-n-iterations 200 (fast). Override via env:
#   SUBJECT=17 ROI=V1 ITERS=200 sbatch verify_cv4.sh
#
# Run this ONLY after confirming no other branch-sensitive jobs are pending
# on the cluster's main checkout (see the deferred-verification note).

set -eo pipefail

LOGFILE="$HOME/logs/cv4_verify_${SLURM_JOB_ID:-local}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

SUBJECT="${SUBJECT:-3}"
ROI="${ROI:-V3AB}"
ITERS="${ITERS:-200}"
bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

echo "Host:    $(hostname)"
echo "Started: $(date)"
echo "Smoke:   sub-${SUBJECT}, roi=${ROI}, iters=${ITERS}"

# CUDA runtime (TF 2.14 / cuDNN 8.7 ABI; CUDA 11.8 spack install).
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

run_arm () {
    local arm="$1"; shift
    echo "=================================================================="
    echo "ARM: ${arm}"
    echo "=================================================================="
    "$PYTHON" -u "$@"
}

# --- MODEL0 (all gains zero) ---
run_arm model0 -m retsupp.modeling.fit_af_prf_cv_v2 "$SUBJECT" \
    --roi "$ROI" --bids-folder "$bids_folder" \
    --sus-hp-sign zero --sus-lp-sign zero \
    --dyn-hp-sign zero --dyn-lp-sign zero --target-gain zero \
    --resolution 50 --max-voxels 500 --p-signal-thr 0.5 \
    --max-n-iterations "$ITERS" \
    --output-subdir af_prf_cv_shiftvsgain/model0

# --- GAIN (all gains free) ---
run_arm gain -m retsupp.modeling.fit_af_prf_cv_v2 "$SUBJECT" \
    --roi "$ROI" --bids-folder "$bids_folder" \
    --sus-hp-sign free --sus-lp-sign free \
    --dyn-hp-sign free --dyn-lp-sign free --target-gain free \
    --resolution 50 --max-voxels 500 --p-signal-thr 0.5 \
    --max-n-iterations "$ITERS" \
    --output-subdir af_prf_cv_shiftvsgain/gain

# --- SHIFT (klein 6-sigma) ---
run_arm shift -m retsupp.modeling.fit_klein_shift_cv "$SUBJECT" \
    --roi "$ROI" --bids-folder "$bids_folder" \
    --max-voxels 500 --p-signal-thr 0.5 \
    --max-n-iterations "$ITERS" \
    --output-subdir af_prf_cv_shiftvsgain/shift

# --- Voxel-id identity + well-formedness assertions ---
echo "=================================================================="
echo "ASSERT: voxel_ids identical across model0/gain/shift; TSVs well-formed"
echo "=================================================================="
"$PYTHON" - "$bids_folder" "$SUBJECT" "$ROI" <<'PY'
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

bids = Path(sys.argv[1]); subject = int(sys.argv[2]); roi = sys.argv[3]
root = bids / 'derivatives' / 'af_prf_cv_shiftvsgain'
arms = ['model0', 'gain', 'shift']
prefix = f'sub-{subject:02d}_roi-{roi}'

ref = None
for arm in arms:
    d = root / arm / f'sub-{subject:02d}'
    r2 = pd.read_csv(d / f'{prefix}_cv-r2.tsv', sep='\t')
    pp = pd.read_csv(d / f'{prefix}_cv-params.tsv', sep='\t')
    meta = json.loads((d / f'{prefix}_meta.json').read_text())
    ids = np.sort(r2['voxel_id'].unique())
    finite = np.isfinite(r2['cv_r2'].to_numpy())
    print(f'{arm}: n_vox={ids.size}, n_rows_r2={len(r2)}, '
          f'finite_cv_r2={int(finite.sum())}/{finite.size}, '
          f'selector={meta.get("selector")}, p_sig={meta.get("p_signal_thr")}, '
          f'param_cols={meta.get("per_voxel_par_cols")}')
    assert finite.any(), f'{arm}: no finite per-voxel CV-R2!'
    assert set(r2['voxel_id']) == set(pp['voxel_id']), \
        f'{arm}: r2/params voxel_id mismatch'
    if ref is None:
        ref = ids
    else:
        assert np.array_equal(ids, ref), f'{arm}: voxel_id != model0 voxel_id'
print('OK: all 3 arms have IDENTICAL voxel_ids and finite CV-R2.')
PY

# --- Aggregator (computes null level, emits per-ROI + sweep tables) ---
echo "=================================================================="
echo "AGGREGATOR"
echo "=================================================================="
"$PYTHON" -u -m retsupp.modeling.aggregate_shiftvsgain_cv \
    --bids-folder "$bids_folder"

echo "Listing emitted aggregate TSVs:"
ls -la "$bids_folder/derivatives/af_prf_cv_shiftvsgain/"cv4_*.tsv

echo "Finished: $(date)"
