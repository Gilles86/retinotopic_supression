#!/bin/bash -l
#SBATCH --job-name=autoflatten
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#
# Cut and flatten one subject's fmriPrep FreeSurfer surfaces with autoflatten.
# Writes {lh,rh}.autoflatten.* into the subject's surf/ folder.
#
# Usage: sbatch --array=3,17,23 retsupp/visualize/pycortex/slurm_jobs/autoflatten.sh

SUBJ=$(printf "sub-%02d" "$SLURM_ARRAY_TASK_ID")
LOGFILE="$HOME/logs/autoflatten_${SUBJ}_${SLURM_JOB_ID}.txt"
exec >"$LOGFILE" 2>&1
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH
export FS_LICENSE=$HOME/freesurfer/license.txt
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep/sourcedata/freesurfer
[ -e "$SUBJECTS_DIR/fsaverage" ] || { echo "No fsaverage in $SUBJECTS_DIR"; exit 1; }

source ~/data/miniforge3/etc/profile.d/conda.sh
conda activate autoflatten

# jaxlib >= 0.11 aborts on the XLA thread flag autoflatten sets (BrokenProcessPool).
SITE=$(mktemp -d)
cat > "$SITE/sitecustomize.py" <<'PY'
import os
import autoflatten.flatten.threading as _t
_configure = _t.configure_threading
def configure_threading(n_threads=None):
    _configure(n_threads)
    os.environ['XLA_FLAGS'] = ' '.join(
        f for f in os.environ.get('XLA_FLAGS', '').split()
        if not f.startswith('--xla_cpu_multi_thread_eigen_thread_count'))
_t.configure_threading = configure_threading
PY
export PYTHONPATH="$SITE"
time autoflatten "$SUBJECTS_DIR/$SUBJ" --parallel --overwrite
