#!/bin/bash
# Live-ish status snapshot for a Snakemake-driven cluster run.
# Shows project context (repo + config + scope), driver state, and
# what's in flight / recently completed grouped meaningfully.
#
# Usage (run on cluster login OR locally via ssh-wrap):
#   bash retsupp/snakemake/status.sh
#   ssh sciencecluster 'bash ~/git/retsupp/retsupp/snakemake/status.sh'

set -eo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SNAKE_DIR="$REPO/retsupp/snakemake"
LOG_DIR="$SNAKE_DIR/logs"
CONFIG="$SNAKE_DIR/config.yaml"

# ---------------------------------------------------------------------
# 1. Project + config context
# ---------------------------------------------------------------------
echo "=== project ==="
echo "  repo:   $REPO"
echo "  config: $CONFIG"

if [ -f "$CONFIG" ]; then
    # Pull config keys via awk (avoid yaml lib dep). Single-line scalars
    # and inline-flow lists only — multi-line block sequences need
    # special handling.
    n_subjects=$(awk '/^subjects:/{flag=1;next} /^[a-z_]+:/{flag=0} flag && /^  -/{c++} END{print c+0}' "$CONFIG")
    n_chunks=$(awk -F': *' '/^n_chunks:/{print $2; exit}' "$CONFIG")
    kind=$(awk -F': *' '/^kind:/{print $2; exit}' "$CONFIG")
    n_models=$(awk '/^models:/{flag=1;next} /^[a-z_]+:/{flag=0} flag && /^  -/{c++} END{print c+0}' "$CONFIG")
    n_neuro_models=$(awk '/^neuropythy_models:/{flag=1;next} /^[a-z_]+:/{flag=0} flag && /^  -/{c++} END{print c+0}' "$CONFIG")
    af_variant_names=$(awk '/^af_variants:/{flag=1;next} /^[a-z_]+:/{flag=0} flag && /^  *- *name:/{sub(/.*name: */, ""); print}' "$CONFIG" | xargs | tr ' ' ',')
    with_af=$(awk -F': *' '/^with_af:/{print $2; exit}' "$CONFIG")

    echo "  subjects=$n_subjects, n_chunks=$n_chunks, kind=$kind, models=$n_models, with_af=$with_af"
    echo "  neuropythy_models=$n_neuro_models, af_variants=[$af_variant_names]"
fi

# ---------------------------------------------------------------------
# 2. Driver state
# ---------------------------------------------------------------------
LATEST_LOG=$(ls -t "$LOG_DIR"/driver_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "no driver log found in $LOG_DIR"
    exit 0
fi
DRIVER_JOBID=$(basename "$LATEST_LOG" .log | sed 's/^driver_//')
DRIVER_STATE=$(squeue -j "$DRIVER_JOBID" -h --format="%T %M" 2>/dev/null || echo "GONE")
echo
echo "=== driver ==="
echo "  job $DRIVER_JOBID  state=$DRIVER_STATE"
echo "  log: $LATEST_LOG"

# Most recent progress line
PROGRESS=$(grep -oE "[0-9]+ of [0-9]+ steps \([0-9]+%\) done" "$LATEST_LOG" | tail -1)
[ -n "$PROGRESS" ] && echo "  progress: $PROGRESS"

# Last 3 lines for context
echo
echo "  --- last 3 driver lines ---"
tail -3 "$LATEST_LOG" | sed 's/^/  /'

# ---------------------------------------------------------------------
# 3. In-flight breakdown by rule + scope
#
# Comment format from the SLURM plugin: rule_<name>_wildcards_<vals>
# where <vals> is the wildcard values joined by underscore in
# Snakefile-declaration order. We extract rule and the leading
# wildcard (almost always sub_pad) and bucket.
# ---------------------------------------------------------------------
echo
echo "=== in-flight by rule (running / pending) ==="
squeue -u "$USER" -h -O "State:14,Comment:80" 2>/dev/null \
    | awk '
{
    state=$1
    full=$2
    sub(/^rule_/, "", full)
    n = split(full, parts, "_wildcards_")
    rule = parts[1]
    wild = (n >= 2 ? parts[2] : "")
    if (full == $2) { rule = "(no-comment)"; wild = "" }
    # First wildcard is usually sub_pad — track unique subs per rule
    n2 = split(wild, w, "_")
    sub_id = (n2 >= 1 ? w[1] : "")
    key = state "|" rule
    counts[key]++
    if (sub_id != "") {
        if (!(key SUBSEP sub_id in seen_sub)) {
            seen_sub[key SUBSEP sub_id] = 1
            sub_count[key]++
        }
    }
}
END {
    for (k in counts) printf "%d\t%s\t%d\n", counts[k], k, (sub_count[k] ? sub_count[k] : 0)
}' \
    | sort -rn \
    | awk -F'\t' -v OFS='' '{
        split($2, sk, "|"); state=sk[1]; rule=sk[2]
        sub_summary = ($3 > 0 ? "  (" $3 " distinct subs)" : "")
        printf "  %-32s %-10s  n=%d%s\n", rule, state, $1, sub_summary
    }' \
    | head -25

# ---------------------------------------------------------------------
# 4. Completed in the last hour, broken down by rule + which subs landed
# ---------------------------------------------------------------------
SINCE=$(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null \
        || date -v-1H +%Y-%m-%dT%H:%M:%S 2>/dev/null)
echo
echo "=== completed in the last hour (since $SINCE) ==="
sacct -u "$USER" -X -S "$SINCE" --format=Comment%80,State -P -n 2>/dev/null \
    | awk -F'|' '$2 == "COMPLETED" {
        full = $1
        sub(/^rule_/, "", full)
        n = split(full, parts, "_wildcards_")
        rule = parts[1]
        wild = (n >= 2 ? parts[2] : "")
        n2 = split(wild, w, "_")
        sub_id = (n2 >= 1 ? w[1] : "")
        counts[rule]++
        if (sub_id != "") {
            if (!(rule SUBSEP sub_id in seen_sub)) {
                seen_sub[rule SUBSEP sub_id] = 1
                sub_count[rule]++
            }
        }
    }
    END {
        for (k in counts) printf "%d\t%s\t%d\n", counts[k], k, (sub_count[k] ? sub_count[k] : 0)
    }' \
    | sort -rn \
    | awk -F'\t' '{
        sub_summary = ($3 > 0 ? "  (" $3 " distinct subs)" : "")
        printf "  %-32s  n=%d%s\n", $2, $1, sub_summary
    }' \
    | head -15

# ---------------------------------------------------------------------
# 5. Recent driver-side errors (last ~50 lines of log only — fast)
# ---------------------------------------------------------------------
echo
echo "=== recent driver errors (last 100 lines of log) ==="
n_err=$(tail -100 "$LATEST_LOG" | grep -cE "Error|Failed|error in" || true)
echo "  $n_err error/failure mentions in tail"
if [ "$n_err" -gt 0 ]; then
    tail -100 "$LATEST_LOG" | grep -E "Error|Failed|error in" | tail -3 | sed 's/^/    /'
fi
