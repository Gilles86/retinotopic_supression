#!/bin/bash
#SBATCH --job-name=rtlatent
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Addendum 6 (notes/rt_power_plan_2026-11-13.md): retsupp/visualize/paper/rt_joint_latent_model.py on a compute node.
# Everything after the script name is passed to the python script, e.g.
#   sbatch --exclude=u24-chaihm0-633,u24-chaiam0-613 --job-name=rtlatent_main retsupp/visualize/paper/slurm_jobs/rt_joint_latent_model.sh \
#       fit --gains notes/data/rt_joint_latent_2026-11-13/trialse_pooled.tsv --out-dir notes/data/rt_joint_latent_2026-11-13 --tag main
set -eo pipefail
LOGFILE="$HOME/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export PYTENSOR_FLAGS="compiledir=$TMPDIR/pytensor"
export XLA_FLAGS="--xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK:-4}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  Args: $*  Started: $(date)"
cd "${REPO:-$HOME/retsupp_plan}"
"$HOME/data/conda/envs/bauer/bin/python" -u retsupp/visualize/paper/rt_joint_latent_model.py "$@"
rm -rf "$TMPDIR"
echo "Finished: $(date)"
