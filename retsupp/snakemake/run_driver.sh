#!/bin/bash
#SBATCH --job-name=prf_snake_driver
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
# lowprio + the only QoS it allows (`normal`) caps at 24h. If the run
# doesn't finish, resubmit — Snakemake's persistence picks up where it
# left off. (For runs that genuinely need > 24h, switch partition to
# `standard` and --qos=medium for 2-day walltime.)
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
