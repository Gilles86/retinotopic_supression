#!/usr/bin/env zsh
# Submit fit_prf_extended.sh for every subject and a chosen model label.
#
# Usage:
#   ./submit_all_prf_extended.zsh <model>            # default subjects
#   ./submit_all_prf_extended.zsh <model> 5 9 14     # explicit subject list

set -euo pipefail

model=${1:-1}
shift || true

if [[ $# -gt 0 ]]; then
    subjects=("$@")
else
    # Read subjects from packaged YAML.
    subjects=($(python -c "from retsupp.utils.data import get_subject_ids; print(' '.join(get_subject_ids()))"))
fi

script_dir="${0:A:h}"
sbatch_script="$script_dir/fit_prf_extended.sh"

for sub in "${subjects[@]}"; do
    # Strip leading zeros for python int parsing (10# in zsh).
    sub_int=$((10#${sub}))
    echo "submitting sub-${sub} model-${model}"
    sbatch "$sbatch_script" "$sub_int" "$model"
done
