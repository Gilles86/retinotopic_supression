#!/bin/bash
#SBATCH --job-name=v1_handoff
#SBATCH --account=hare.econ.uzh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=%j_handoff.log

# Build V1 decoding handoff NPZ for one subject. Usage:
#   sbatch --export=ALL,SUB=15,MODEL=4 build_v1_handoff.sh

set -eo pipefail
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

$HOME/data/conda/envs/retsupp_cuda/bin/python - <<PY
import sys, time, os
sys.path.insert(0, "/home/gdehol/git/retsupp")
import numpy as np
from retsupp.utils.data import Subject

t0 = time.time()
SUB = int(os.environ["SUB"]); MODEL = int(os.environ.get("MODEL", 4))
sub = Subject(SUB, "/shares/zne.uzh/gdehol/ds-retsupp")
pars = sub.get_warmstart_pars(model=MODEL, roi="V1")
bold = sub.get_concatenated_bold(voxel_idx=pars["voxel_idx"].to_numpy())
print(f"[{time.time()-t0:.1f}s] pars={pars.shape}  bold={bold.shape}")
os.makedirs("/shares/zne.uzh/gdehol/ds-retsupp/derivatives/v1_decode_handoffs", exist_ok=True)
out = f"/shares/zne.uzh/gdehol/ds-retsupp/derivatives/v1_decode_handoffs/sub-{SUB:02d}_m{MODEL}_V1.npz"
np.savez(out,
    x=pars.x.values, y=pars.y.values, sd=pars.sd.values,
    baseline=pars.baseline.values, amplitude=pars.amplitude.values,
    srf_amplitude=pars.srf_amplitude.values,
    srf_size=pars.srf_size.values,
    hrf_delay=pars.hrf_delay.values,
    hrf_dispersion=pars.hrf_dispersion.values,
    r2=pars.r2.values, hemi=pars.hemi.values,
    voxel_idx=pars.voxel_idx.values, bold=bold,
    tr=1.6, resolution=50, grid_radius=5.0, aperture_radius=3.17,
    subject_id=SUB, model=f"m{MODEL} (warmstart V1)")
print(f"[{time.time()-t0:.1f}s] wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
PY
