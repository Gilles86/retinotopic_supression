#!/bin/bash
# One-shot experiment: how long does sub-02 m1 (full paradigm) take
# in different chunk + hardware configs? All writes go to
# derivatives/prf.bench/model1/sub-02/ so canonical fits are safe.
#
# Submits 4 configs:
#   1. GPU monolithic         (1 job, ~40 min — same as fit_prf_l4.sh default)
#   2. GPU chunked N=10       (10 jobs, each ~5-10 min if data load + GD scale)
#   3. CPU chunked N=10       (10 jobs, each ~longer; CPU is slower per voxel)
#   4. CPU chunked N=54       (54 jobs, the existing fit_prf_chunked default)
#
# Run on the cluster:
#   bash ~/git/retsupp/retsupp/modeling/slurm_jobs/bench_chunked.sh
#
# After all jobs land, collect per-task wallclock from sacct.

set -eo pipefail
SUB=2
MODEL=1
KIND=full

SCRIPT_GPU=$HOME/git/retsupp/retsupp/modeling/slurm_jobs/fit_prf_l4.sh
SCRIPT_CPU=$HOME/git/retsupp/retsupp/modeling/slurm_jobs/fit_prf_chunked.sh

# Tight walltime per config — higher priority bucket = faster
# dispatch. Generous defaults in the script (4h for GPU, 1h for CPU)
# bury us at the bottom of the queue.
echo "=== bench config 1: GPU monolithic ==="
J1=$(sbatch --array=$SUB --time=01:00:00 \
    --export=ALL,MODEL=$MODEL,KIND=$KIND,OUTPUT_SUFFIX=bench.gpu1x \
    "$SCRIPT_GPU" | awk '{print $4}')
echo "  -> $J1"

echo "=== bench config 2: GPU chunked N=10 ==="
# fit_prf_l4.sh uses SLURM_ARRAY_TASK_ID as subject; spawn 10 jobs.
J2_IDS=""
for c in $(seq 0 9); do
    J=$(sbatch --array=$SUB --time=00:20:00 \
        --export=ALL,MODEL=$MODEL,KIND=$KIND,OUTPUT_SUFFIX=bench.gpu10x,CHUNK_INDEX=$c,N_CHUNKS=10 \
        "$SCRIPT_GPU" | awk '{print $4}')
    J2_IDS="${J2_IDS}${J}_${SUB} "
done
echo "  -> $J2_IDS"

echo "=== bench config 3: CPU chunked N=10 ==="
# fit_prf_chunked.sh maps SLURM_ARRAY_TASK_ID = sub_idx*N_CHUNKS + chunk_idx + 1.
# For sub-02 only with N=10, array indices = 11..20  (sub_idx=1 → ((sub-1)*N_CHUNKS+1)..(sub*N_CHUNKS))
# OUTPUT_SUFFIX isn't supported by fit_prf_chunked.sh; we'll bench-merge separately.
# So we just submit one array; output lands under derivatives/prf/model1/sub-02/chunks/
# (canonical). We'll inspect sacct timing only, not the NIfTIs.
J3=$(sbatch --array=11-20 --time=00:40:00 \
    --export=ALL,MODEL=$MODEL,KIND=$KIND,N_CHUNKS=10,N_SUBS=2 \
    "$SCRIPT_CPU" | awk '{print $4}')
echo "  -> $J3"

echo "=== bench config 4: CPU chunked N=54 ==="
J4=$(sbatch --array=55-108 --time=00:20:00 \
    --export=ALL,MODEL=$MODEL,KIND=$KIND,N_CHUNKS=54,N_SUBS=2 \
    "$SCRIPT_CPU" | awk '{print $4}')
echo "  -> $J4"

echo
echo "Once all done, collect timing with:"
echo "  sacct -j $J1,$J3,$J4 --format=JobID,State,Elapsed,JobName -P"
echo "  and for config 2: sacct -j $(echo $J2_IDS | tr ' ' ',') --format=JobID,State,Elapsed -P"
