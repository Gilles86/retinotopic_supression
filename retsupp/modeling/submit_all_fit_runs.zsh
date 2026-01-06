#!/bin/bash
# Submit all fit runs for subjects 1-29, sessions 1-2, runs 1-6, model 4

# Define arrays
subjects=( $(seq 1 29) )
sessions=(1 2)
runs=(1 2 3 4 5 6)
models=(4)

# Loop through all combinations
for subject in "${subjects[@]}"; do
  for session in "${sessions[@]}"; do
    for run in "${runs[@]}"; do
      for model in "${models[@]}"; do
        echo "Submitting job: Subject $subject, Session $session, Run $run, Model $model"
        sbatch $HOME/git/retsupp/retsupp/modeling/fit_run.sh "$subject" "$session" "$run" "$model"
      done
    done
  done
done

