#!/bin/bash
# Live-ish status snapshot for the retsupp Snakemake driver.
# Usage (from cluster login OR locally via ssh-wrap):
#   bash retsupp/snakemake/status.sh
#   ssh sciencecluster 'bash ~/git/retsupp/retsupp/snakemake/status.sh'

set -eo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOG_DIR="$REPO/retsupp/snakemake/logs"

# Latest driver log
LATEST_LOG=$(ls -t "$LOG_DIR"/driver_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "no driver log found in $LOG_DIR"
    exit 1
fi
echo "=== driver log: $LATEST_LOG ==="

# Driver process state
DRIVER_JOBID=$(basename "$LATEST_LOG" .log | sed 's/^driver_//')
DRIVER_STATE=$(squeue -j "$DRIVER_JOBID" -h --format="%T" 2>/dev/null || echo "GONE")
echo "  driver job: $DRIVER_JOBID  state=$DRIVER_STATE"

# Most recent progress line (Snakemake prints "X of Y steps (Z%) done")
PROGRESS=$(grep -oE "[0-9]+ of [0-9]+ steps \([0-9]+%\) done" "$LATEST_LOG" | tail -1)
[ -n "$PROGRESS" ] && echo "  progress:   $PROGRESS"

# Last 3 dispatch lines for context
echo
echo "=== last 5 driver lines ==="
tail -5 "$LATEST_LOG"

# In-flight jobs by rule (parsed from --comment, which the SLURM plugin
# fills with "rule_<name>_wildcards_<wildcards>" — much more useful than
# the UUID job-name).
echo
echo "=== in-flight jobs by rule + state ==="
squeue -u "$USER" -h -O "State:14,Comment:80" 2>/dev/null \
    | awk '{
        state=$1
        # Drop "rule_" prefix and "_wildcards_<...>" trailer for grouping
        rule=$2
        sub(/^rule_/, "", rule)
        sub(/_wildcards_.*/, "", rule)
        # the plugin can produce empty comment on raw bash submissions
        if (rule == "" || rule == $2) rule = "(other)"
        counts[state "|" rule]++
    }
    END {
        for (k in counts) print counts[k], k
    }' \
    | sort -rn | head -20 \
    | awk -F'|' '{printf "  %-50s %s\n", $2, $1}'

# Recently completed (last hour) by rule
echo
echo "=== completed in the last hour by rule ==="
sacct -u "$USER" -X -S "$(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-1H +%Y-%m-%dT%H:%M:%S)" \
    --format=Comment%80,State -P -n 2>/dev/null \
    | awk -F'|' '$2 == "COMPLETED" {
        rule=$1
        sub(/^rule_/, "", rule)
        sub(/_wildcards_.*/, "", rule)
        counts[rule]++
    }
    END {
        for (k in counts) printf "  %-50s %d\n", k, counts[k]
    }' \
    | sort -k2 -rn | head -10
