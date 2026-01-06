#!/bin/bash
#SBATCH --job-name=nordic_job            # Job name
#SBATCH --output=/home/gdehol/logs/nordic_job_%j.log  # Standard output and error log
#SBATCH --ntasks=32                      # Number of cores
#SBATCH --time=02:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=64gb                       # Memory limit
#SBATCH --constraint=INTEL

source /etc/profile.d/lmod.sh

# Load the necessary modules (adjust as needed for your environment)
module load matlab

# Initialize variables
BASE_PATH="/shares/zne.uzh/gdehol/ds-retsupp"
SUBJECT=""
RUN=""
SESSION=""

# Check for optional arguments
while getopts b:s:r:e: flag
do
    case "${flag}" in
        b) BASE_PATH=${OPTARG};;
        s) SUBJECT=${OPTARG};;
        r) RUN=${OPTARG};;
        e) SESSION=${OPTARG};;
    esac
done

# Check if BASE_PATH and SESSION are set and construct the MATLAB command accordingly
if [ -z "$BASE_PATH" ]
then
    if [ -z "$SESSION" ]
    then
        MATLAB_CMD="run_nordic('$SUBJECT', $RUN); exit"
    else
        MATLAB_CMD="run_nordic('$SUBJECT', $RUN, '', $SESSION); exit"
    fi
else
    if [ -z "$SESSION" ]
    then
        MATLAB_CMD="run_nordic('$SUBJECT', $RUN, '$BASE_PATH'); exit"
    else
        MATLAB_CMD="run_nordic('$SUBJECT', $RUN, '$BASE_PATH', $SESSION); exit"
    fi
fi

# Run MATLAB
matlab -nodisplay -r "$MATLAB_CMD"
