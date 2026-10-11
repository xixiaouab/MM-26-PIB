#!/bin/bash
# Extract the FGVC/HTA archives and write manifests, one process per dataset, on one Frontier node.
#   sbatch --export=ALL,NAMES="flowers dogs cub" scripts/prepare_data.sbatch
#SBATCH -A LRN036
#SBATCH -J pib-prep
#SBATCH -p batch
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -o /lustre/orion/csc662/proj-shared/xxiao/PIB/logs/%x-%j.out
set -euo pipefail
source /lustre/orion/csc662/proj-shared/xxiao/PIB/code/scripts/frontier_env.sh
cd "$PIB_ROOT/code"
D=$PIB_ROOT/data
NAMES=${NAMES:-"flowers dogs cub nabirds cars dtd svhn gtsrb food101 cifar10 cifar100"}
pids=()
for name in $NAMES; do
  python scripts/prepare_fgvc_hta.py --raw "$D/raw" --images "$D/images" --manifests "$D/manifests" \
    --workers 8 "$name" > "$PIB_ROOT/logs/prep_$name.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for name in $NAMES; do echo "== $name"; grep -E "classes=|Error|Traceback" "$PIB_ROOT/logs/prep_$name.log" || true; done
exit $status
