#!/bin/bash -l
#SBATCH --job-name=fmriprep_retsupp
#SBATCH --output=/home/gdehol/logs/retsupp_fmriprep_%A-%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
# 8 CPUs: measured ~1.07x slower than 16 at about half the CPU cost; memory sized from real
# (anon) use, not sacct MaxRSS. See the fmriprep skill, references/resources.md.

# Initialize Lmod
source /etc/profile.d/lmod.sh

# Load Apptainer module
module load apptainer/1.4.1

# Set environment variables (Apptainer-compatible)
export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Define the path to the bids_filter.json inside the container
BIDS_FILTER_FILE="/bids_input/bids_filter.json"

# Run the container using Apptainer
# Tell fMRIPrep its allocation (otherwise nipype plans with the whole node's RAM and cores).
NCPU=${SLURM_CPUS_PER_TASK:-8}
MEM_MB=$(( ${SLURM_MEM_PER_NODE:-65536} - 8000 ))
STDOUT_COPY="/scratch/$USER/fmriprep_stdout_sub-${PARTICIPANT_LABEL}_${SLURM_JOB_ID:-manual}.log"

set -o pipefail
apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-retsupp:/data \
  -B /scratch/gdehol:/workflow \
  -B ${PWD}:/bids_input \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.3 \
    /data /data/derivatives/fmriprep participant \
  --participant_label $PARTICIPANT_LABEL \
  --output-spaces T1w fsnative \
  --dummy-scans 4 \
  --skip_bids_validation \
  -w /workflow \
  --nthreads "$NCPU" \
  --omp-nthreads "$NCPU" \
  --mem-mb "$MEM_MB" \
  --low-mem \
  --bids-filter-file $BIDS_FILTER_FILE \
  --force syn-sdc 2>&1 | tee "$STDOUT_COPY"
RC=${PIPESTATUS[0]}

# apptainer >= 1.4 can exit 1 after a clean run (value_capture, 2026-11-14: all 12 subjects),
# and fMRIPrep writes an html on failures too, so neither is proof of success. fMRIPrep's own
# last line, printed only after a successful run, is. See the fmriprep skill.
if [[ $RC -ne 0 ]]; then
    if grep -q "fMRIPrep finished successfully" "$STDOUT_COPY"; then
        echo "apptainer exited $RC but fMRIPrep reported success: treating as a clean run."
    else
        echo "apptainer exited $RC and fMRIPrep did not report success: failed."
        exit $RC
    fi
fi

