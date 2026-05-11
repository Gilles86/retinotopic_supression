#!/bin/bash
#SBATCH --job-name=neuropythy_sub16
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Recovery wrapper for sub-16 specifically. The canonical freesurfer
# recon at sourcedata/freesurfer/sub-16 has different vertex counts
# than the prf .gii surface fits (lh.white: 261188 vs .gii: 261790;
# rh.white: 262394 vs .gii: 261900) — neuropythy register_retinotopy
# bails on the LH predicted-mesh export with "length should be …"
# because the .mgz inputs don't fit the registered mesh.
#
# A second freesurfer recon for the same subject lives at
# sourcedata/freesurfer/sub-16_ses-1 (mtime 2025-11-03, newer than the
# sub-16 recon's 2025-10-28). Its vertex counts MATCH the prf .gii
# inputs and its orig.mgz is byte-identical to sub-16/mri/orig.mgz.
# So we run register_retinotopy against sub-16_ses-1, then copy the
# resulting volume `inferred_*.mgz` outputs back into sub-16/mri/ so
# that downstream consumers (Subject.get_retinotopic_roi via
# inferred_varea.mgz) keep working unchanged.
#
# Usage:
#   sbatch retsupp/neuropythy/slurm_jobs/register_retinotopy_sub16.sh

set -euo pipefail

LOGFILE="$HOME/logs/neuropythy_sub16_recovery_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host: $(hostname) | sub-16 recovery via sub-16_ses-1 freesurfer dir"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_neuropythy
export PYTHONUNBUFFERED=1

export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt
[[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]] && \
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep/sourcedata/freesurfer
echo "FREESURFER_HOME=${FREESURFER_HOME}"
echo "SUBJECTS_DIR=${SUBJECTS_DIR}"
echo "FS version: $(mri_surf2surf --version 2>&1 | head -1)"

PYTHON="$HOME/data/conda/envs/retsupp_neuropythy/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/neuropythy/register_retinotopy.py" \
    16 \
    --bids_dir /shares/zne.uzh/gdehol/ds-retsupp \
    --fs-subject sub-16_ses-1

echo ""
echo "=== Copying inferred_*.mgz volumes back into sub-16/mri/ ==="
SRC="${SUBJECTS_DIR}/sub-16_ses-1/mri"
DST="${SUBJECTS_DIR}/sub-16/mri"
for f in inferred_angle.mgz inferred_eccen.mgz inferred_sigma.mgz inferred_varea.mgz; do
    if [[ -f "${SRC}/${f}" ]]; then
        cp -v "${SRC}/${f}" "${DST}/${f}"
    else
        echo "WARN: ${SRC}/${f} missing — neuropythy may have not produced it."
    fi
done

# Also propagate surface inferred files into sub-16_ses-1/surf only —
# we deliberately do NOT copy these into sub-16/surf/ since the meshes
# have different vertex counts.  Downstream code that needs surface
# inferred outputs (get_inferred_retinotopy_surface) is rarely used and
# will produce wrong results for sub-16 if pointed at sub-16/surf — use
# sub-16_ses-1 if such code is ever run on this subject.

echo ""
echo "Finished: $(date)"
