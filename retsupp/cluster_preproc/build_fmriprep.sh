#!/bin/bash
#SBATCH --job-name=build_fmriprep
#SBATCH --output=/home/gdehol/logs/build_fmriprep.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
source /etc/profile.d/lmod.sh
module load singularityce/3.10.2
singularity build --sandbox /shares/zne.uzh/containers/fmriprep-24.0.1 docker://nipreps/fmriprep:24.0.1
