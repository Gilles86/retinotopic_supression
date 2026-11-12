#!/bin/bash -l
#SBATCH --job-name=decode_simfit
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# Fitted-model prediction of the decoding readouts (simulate_decode.py --mode fitted): one task = one subject × session;
# per run and ROI (V1-V3) the polished m7_full joint fit (model) and the same with its gains at 0 (control), plus one
# phase-randomised residual noise draw, decoded with the fold pRFs as in Fig. 6 (free and outline decoders).
# Task index -> subject = (i - 1) / 2 + 1, session = (i - 1) % 2 + 1; sub-03 ... sub-30 = 5-60.
#   cd ~/retsupp_plan && sbatch --array=5-60 retsupp/decode/slurm_jobs/simulate_decode_fitted.sh
#   LADDER_ROOT=<root> HRF_TABLE=<tsv> EXTRA_ARGS="--run 1" sbatch ...

set -eo pipefail
idx0=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array} - 1 ))
subject=$(( idx0 / 2 + 1 ))
session=$(( idx0 % 2 + 1 ))
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives
LADDER_ROOT="${LADDER_ROOT:-$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished}"
HRF_TABLE="${HRF_TABLE:-$D/paper_2026-11/roi_hrf_dog_final2.tsv}"
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/decode_simfit_sub-${subject}_ses-${session}_${SLURM_JOB_ID}.txt" 2>&1
echo "Host $(hostname)  sub-${subject} ses-${session}  $(date)"
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}" TF_NUM_INTEROP_THREADS=2
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.decode.simulate_decode "$subject" --mode fitted \
    --session "$session" --prf-source jointhrf --ladder-root "$LADDER_ROOT" --hrf-table "$HRF_TABLE" ${EXTRA_ARGS:-}
echo "Finished $(date)"
