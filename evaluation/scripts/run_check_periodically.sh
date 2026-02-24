#!/bin/bash
set -e

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

# Common parameters (aligned with run_longbench_v2_eval.sh)
LIB="dify"
VERSION_DIR="longbench_v2_dify_0129"

echo "Starting periodic check for LIB=$LIB, VERSION_DIR=$VERSION_DIR"
echo "Check interval: 1 hour"

while true; do
    echo "----------------------------------------------------------------"
    echo "Running longbench_v2_check_files.py at $(date)..."

    python -m evaluation.scripts.longbench_v2.longbench_v2_check_files \
      --lib "$LIB" \
      --version-dir "$VERSION_DIR"

    echo "Finished check at $(date). Sleeping for 1 hour..."
    sleep 3600
done
