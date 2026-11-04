#!/bin/bash
# Whole-brain m6 (DN + flexible HRF) with braincoder's flexible-HRF fix, per subject:
# chunks on CPU -> merge -> surface sampling. Writes to derivatives/prf.hrffix/; the
# May 2026 fits in prf/model6 are left alone. Initialises from prf/model5.
#
# Usage, from the repo root on the login node:
#   bash retsupp/modeling/slurm_jobs/submit_m6_refit.sh 3
#   bash retsupp/modeling/slurm_jobs/submit_m6_refit.sh $(seq 1 30)
set -euo pipefail
sb() { sbatch "$@" | awk '{print $4}'; }
export OUTPUT_SUFFIX=hrffix MODEL=6 N_CHUNKS=10 KIND=full
for s in "$@"; do
    C=$(SUBJECT=$s sb --partition=standard --mem=40G --time=04:30:00 --array=1-$N_CHUNKS retsupp/modeling/slurm_jobs/fit_prf_cpu_chunked.sh)
    M=$(sb --partition=standard --time=00:10:00 --array=$s --dependency=afterok:$C retsupp/modeling/slurm_jobs/merge_prf_chunks.sh)
    S=$(sb --partition=standard --time=00:20:00 --array=$s --dependency=afterok:$M retsupp/surface/slurm_jobs/sample_prf_to_surface.sh)
    echo "sub-$s chunks $C merge $M surface $S"
done
