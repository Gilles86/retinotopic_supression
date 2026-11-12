#!/bin/bash -l
#SBATCH --job-name=sim_prf_size_floor
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G          # canary: 5 GB at res 50, >16 GB (killed) at res 150
#SBATCH --time=01:00:00    # canary: 3 min at res 50 (8 CPUs)
#
# Parameter recovery of small pRFs (cv/sim_prf_size_floor.py). One array task per (family, R², condition):
# task id = 100*family_idx + 10*r2_idx + condition_idx, family_idx 0 = gauss, 1 = dog; r2_idx 0 = .35, 1 = .15;
# condition_idx into CONDS below.
#   cd ~/retsupp_plan && sbatch --array=100,107 retsupp/modeling/cv/slurm_jobs/sim_prf_size_floor.sh   # canary

set -eo pipefail
FAMS=(gauss dog)
R2S=(.35 .15)
CONDS=(paper50 paper100 paper150 grid50 grid100 free50 free100 free150)
id=${SLURM_ARRAY_TASK_ID}
fam=${FAMS[$(( id / 100 ))]}
r2=${R2S[$(( (id / 10) % 10 ))]}
cond=${CONDS[$(( id % 10 ))]}
OUT="${OUT:-/shares/zne.uzh/gdehol/ds-retsupp/derivatives/sim_prf_size_floor_2026-11-12}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/sim_prf_size_floor_${fam}_r2-${r2}_${cond}_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | ${fam} r2 ${r2} ${cond} | job ${SLURM_JOB_ID} | $(date)"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1 KERAS_BACKEND=tensorflow
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}" TF_NUM_INTEROP_THREADS=2
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.cv.sim_prf_size_floor \
    --family "$fam" --r2 "$r2" --condition "$cond" --out-dir "$OUT"
echo "Finished: $(date)"
