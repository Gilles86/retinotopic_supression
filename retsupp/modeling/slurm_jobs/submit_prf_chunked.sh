#!/bin/bash
# Submit chunked PRF fit with --mem auto-matched to per-task voxel count.
#
# Usage:
#   bash submit_prf_chunked.sh [MODEL] [N_CHUNKS] [N_SUBS]
#
# Defaults: MODEL=1, N_CHUNKS=54, N_SUBS=30.
#
# Memory model (empirical, Gaussian model 1, T=3096):
#   base = ~6 GB (BOLD + paradigm + masker)
#   per-voxel-activation = ~1.2 GB per 1000 voxels
#   peak ~ base + 1.2 * vox_per_task / 1000
# We round up + 50% safety; DoG/DN need ~1.5x — bump --mem manually
# for those models if you see OOMs.

set -euo pipefail

MODEL="${1:-1}"
N_CHUNKS="${2:-54}"
N_SUBS="${3:-30}"

# Voxel counts across subjects vary substantially (sub-11=245k -> sub-08=376k).
# Size mem for the LARGEST subject + headroom so all tasks fit.
MAX_TOTAL_VOX=400000   # > observed max 376475
VOX_PER_CHUNK=$(( MAX_TOTAL_VOX / N_CHUNKS ))

# Memory: 6 GB base + 1.2 GB per 1000 voxels, +70% safety, rounded up.
MEM_GB=$(awk -v v=$VOX_PER_CHUNK 'BEGIN{
    raw = 6 + 1.2 * v / 1000;
    safe = raw * 1.7;
    printf "%d\n", (safe == int(safe) ? safe : int(safe) + 1)
}')
# Bump model 4 (DoG flex HRF) by 1.5x; models 5/6 (DN) by 2x.
case "$MODEL" in
    2|4) MEM_GB=$(( MEM_GB * 3 / 2 )) ;;
    5|6) MEM_GB=$(( MEM_GB * 2 )) ;;
esac

ARRAY_SIZE=$(( N_SUBS * N_CHUNKS ))
# Throttle: max concurrent running tasks. Caps simultaneous NFS profile
# reads at job start ("user env retrieval failed" mitigation). 50 is a
# safe default for our cluster.
THROTTLE="${THROTTLE:-50}"

echo "Submitting: MODEL=$MODEL, N_CHUNKS=$N_CHUNKS, N_SUBS=$N_SUBS"
echo "  per-task: ~$VOX_PER_CHUNK voxels, --mem=${MEM_GB}G, --cpus=16"
echo "  array: 1-$ARRAY_SIZE%$THROTTLE"

sbatch --array="1-$ARRAY_SIZE%$THROTTLE" \
       --mem="${MEM_GB}G" \
       --export="ALL,MODEL=$MODEL,N_CHUNKS=$N_CHUNKS,N_SUBS=$N_SUBS" \
       "$(dirname "$0")/fit_prf_chunked.sh"
