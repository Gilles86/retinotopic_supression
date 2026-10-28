#!/bin/bash
# Submit the nested gain-field model comparison (one SLURM array per model).
#
# Components (Theeuwes-style priority map, decomposed):
#   T   target gain (dynamic, any ring location)              -> g_T_dyn
#   D   distractor gain, location-blind (dynamic)             -> g_HP_dyn := g_LP_dyn
#   S   sustained gain, location-blind (all 4 ring locations) -> g_HP := g_LP
#   dD  dynamic statistical learning (HP != LP)               -> g_HP_dyn free
#   dS  sustained statistical learning (HP != LP)             -> g_HP free
# Every model except m00 includes the pRF + rectangle search items with a free
# item-drive scale (the stimulus model); m0 has no gain field at all; m00 is the
# vanilla pRF on the bar-only stimulus.
#
# Usage (from ~/git/retsupp on the cluster login node):
#   bash retsupp/modeling/slurm_jobs/submit_cv_ladder.sh            # 9-model ladder, 30 subs
#   PILOT=1 bash retsupp/modeling/slurm_jobs/submit_cv_ladder.sh    # 3 subs x 11 ROIs
#   CV_SCHEME=run3 bash .../submit_cv_ladder.sh                     # 3 folds, 1 run per HP held out
#   CPU=1 bash .../submit_cv_ladder.sh                              # CPU, 8 cores/task (no GPU); PARTITION=lowprio to use idle lowprio nodes
#   MODELS="m0_prf m7_full" bash .../submit_cv_ladder.sh            # subset
#   EXTRA=1 bash .../submit_cv_ladder.sh                            # + drop-one / item-scale controls

set -euo pipefail
cd "$(dirname "$0")"

#             name            SUS_HP SUS_LP DYN_HP DYN_LP TGT   ITEM_SCALE ITEM_SHAPE
LADDER=(
    "m00_bar              zero   zero   zero   zero   zero  none  none"
    "m0_prf               zero   zero   zero   zero   zero  free  rectangle"
    "m1_T                 zero   zero   zero   zero   free  free  rectangle"
    "m2_D                 zero   zero   tied   free   zero  free  rectangle"
    "m3_S                 tied   free   zero   zero   zero  free  rectangle"
    "m3b_TD               zero   zero   tied   free   free  free  rectangle"
    "m4_TDS               tied   free   tied   free   free  free  rectangle"
    "m5_TDS_dS            free   free   tied   free   free  free  rectangle"
    "m6_TDS_dD            tied   free   free   free   free  free  rectangle"
    "m7_full              free   free   free   free   free  free  rectangle"
)
EXTRAS=(
    "m7_full_minusT       free   free   free   free   zero  free  rectangle"
    "m7_full_minusD       free   free   zero   zero   free  free  rectangle"
    "m7_full_minusS       zero   zero   free   free   free  free  rectangle"
    "m0_prf_itemsfixed    zero   zero   zero   zero   zero  fixed rectangle"
    "m7_full_itemsfixed   free   free   free   free   free  fixed rectangle"
)

SPECS=("${LADDER[@]}")
if [[ "${EXTRA:-0}" == "1" ]]; then
    SPECS+=("${EXTRAS[@]}")
fi

PILOT="${PILOT:-0}"
if [[ "$PILOT" == "1" ]]; then N_TASKS=33; else N_TASKS=330; fi
THROTTLE="${THROTTLE:-150}"
CV_SCHEME="${CV_SCHEME:-loco}"
if [[ "${CPU:-0}" == "1" ]]; then
    JOB_SCRIPT=fit_cv_ladder_cpu.sh
    RES=(--cpus-per-task=8 --partition="${PARTITION:-standard}")
else
    JOB_SCRIPT=fit_cv_ladder.sh
    RES=()
fi

for spec in "${SPECS[@]}"; do
    read -r name sus_hp sus_lp dyn_hp dyn_lp tgt item_scale item_shape <<< "$spec"
    if [[ -n "${MODELS:-}" ]] && [[ " ${MODELS} " != *" ${name} "* ]]; then
        continue
    fi
    jid=$(sbatch --parsable --array=1-${N_TASKS}%${THROTTLE} "${RES[@]}" \
        --export=ALL,MODEL="$name",SUS_HP="$sus_hp",SUS_LP="$sus_lp",DYN_HP="$dyn_hp",DYN_LP="$dyn_lp",TGT="$tgt",ITEM_SHAPE="$item_shape",ITEM_SCALE="$item_scale",PILOT="$PILOT",CV_SCHEME="$CV_SCHEME" \
        "$JOB_SCRIPT")
    echo "${CV_SCHEME} ${name}: job ${jid} (${N_TASKS} tasks)"
done
