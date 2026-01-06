#!/bin/bash
#SBATCH --job-name=fmriprep_retinonumeral
#SBATCH --output=/home/gdehol/logs/retsupp_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00

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
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --bids-filter-file $BIDS_FILTER_FILE \
  --force syn-sdc
