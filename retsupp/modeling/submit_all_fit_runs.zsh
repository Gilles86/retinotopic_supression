#!/bin/zsh

subjects=(1 2 3 5 10 11 12 13)
sessions=(1 2)
runs=(1 2 3 4 5 6)
models=(1 2 3 4)

for subject in $subjects; do
  for session in $sessions; do
    for run in $runs; do
      for model in $models; do
        sbatch /Users/gdehol/git/retsupp/retsupp/modeling/fit_run.sh $subject $session $run $model
      done
    done
  done
done
