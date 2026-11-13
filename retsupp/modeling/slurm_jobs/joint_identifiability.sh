#!/bin/bash
#SBATCH --job-name=joint_ident
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#
# Identifiability of the sustained LP gain with free pRFs (joint_identifiability.py): dump / fit / profile.
# Pass settings in the environment (no --export lists):
#   STAGE=dump SUBJECT=3 ROIS="V1 TO IPS" sbatch retsupp/modeling/slurm_jobs/joint_identifiability.sh
#   STAGE=fit SUBJECT=3 ROIS=IPS MODEL=m7_full sbatch ...
#   STAGE=profile SUBJECT=3 ROIS=IPS MODEL=m7_full PARAM=g_LP [SPLITS="all"] [GAIN_LO=-1] sbatch ...
#   STAGE=hessian SUBJECT=3 ROIS=IPS MODEL=m7_full sbatch ...
set -eo pipefail
STAGE="${STAGE:?}"; SUBJECT="${SUBJECT:-3}"; ROIS="${ROIS:-IPS}"; MODEL="${MODEL:-m7_full}"; PARAM="${PARAM:-g_LP}"
LOGFILE="$HOME/logs/joint_ident_${STAGE}_sub-$(printf %02d "$SUBJECT")_${ROIS// /-}_${MODEL}_${PARAM}${GAIN_LO:+_glo$GAIN_LO}${ZEROSUM:+_zerosum}${SPLITS:+_${SPLITS// /-}}_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" name="joint_${STAGE}_${MODEL}_${ROIS// /-}" 2>/dev/null || true
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | ${STAGE} sub-${SUBJECT} ${ROIS} ${MODEL} ${PARAM} | $(date)"
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}" TF_NUM_INTEROP_THREADS=2 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
CODE_DIR="$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.joint_identifiability "$STAGE" "$SUBJECT" \
    --roi $ROIS --model "$MODEL" --param "$PARAM" ${SPLITS:+--splits $SPLITS} ${GRID:+--grid $GRID} \
    ${MAX_ITER:+--max-iter $MAX_ITER} ${FTOL:+--ftol $FTOL} ${GAIN_LO:+--gain-lo $GAIN_LO} ${FROM_START:+--from-start} ${ZEROSUM:+--zerosum}
echo "Finished: $(date)"
