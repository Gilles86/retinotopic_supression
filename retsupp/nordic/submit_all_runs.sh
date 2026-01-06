#!/bin/bash

# Check if subject argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 subject_id [session_id]"
    exit 1
fi

SUBJECT=$1
SESSION=""

# Check if session argument is provided
if [ ! -z "$2" ]; then
    SESSION=$2
fi

# Submit jobs for 6 runs with 10-second delay
for RUN in {1..6}; do
    if [ -z "$SESSION" ]; then
        sbatch submit_nordic_job.sh -s $SUBJECT -r $RUN
    else
        sbatch submit_nordic_job.sh -s $SUBJECT -r $RUN -e $SESSION
    fi
    # sleep 10
done
