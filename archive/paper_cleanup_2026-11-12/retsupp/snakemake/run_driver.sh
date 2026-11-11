#!/bin/bash
#SBATCH --job-name=prf_snake_driver
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
# `standard` + `medium` qos = 2-day walltime + different priority
# bucket than lowprio. Was on lowprio earlier but lowprio fairshare
# gets wrecked when child jobs (PRF chunks, AF) saturate it; the
# driver itself then sits PD on `Priority` indefinitely. Driver is
# small (2 CPU, 8G), so standard is a fine fit.
# If 48h isn't enough, resubmit — Snakemake's .snakemake/ persistence
# picks up where it left off.
#SBATCH --output=/home/gdehol/git/retsupp/retsupp/snakemake/logs/driver_%j.log

# Snakemake driver for the retsupp PRF + AF pipeline. Runs as a SLURM
# job so it stays under a compute-node thread/process limit (the
# login node has ulimit -u = 256, which Snakemake exceeds at >100
# concurrent tracked jobs — see CLAUDE.md / snakemake README).
#
# Submit with:
#   ssh sciencecluster 'cd ~/git/retsupp && sbatch retsupp/snakemake/run_driver.sh'
#
# Cancel via:
#   scancel <driver_jobid>
# (SLURM cancellation also kills the children the driver has submitted
# IFF you use scancel --signal=TERM <jobid>; default scancel only
# stops the driver process, the running sub-jobs continue. To stop
# sub-jobs separately: see CLAUDE.md "Cancelling chains".)

set -eo pipefail

echo "Host:    $(hostname)"
echo "JobID:   ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "ulimit -u: $(ulimit -u)"

cd "$HOME/git/retsupp"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_snake

# Allow sbatch invocation from inside this SLURM job. UZH sciencecluster
# permits this on lowprio.
exec snakemake \
    --snakefile retsupp/snakemake/Snakefile \
    --workflow-profile retsupp/snakemake/profile \
    --configfile retsupp/snakemake/config.yaml \
    --rerun-incomplete
