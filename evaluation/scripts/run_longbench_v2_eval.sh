#!/bin/bash
set -e

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

# Common parameters
LIB="fastgpt"
WORKERS=5
TOPK=30
ADD_MODE="fine"
SEARCH_MODE="fast"
VERSION_DIR="longbench_v2_fastgpt_0114"
ASYNC_MODE="sync"
CHAT_MODEL="gpt-4o-mini"
#CHAT_MODEL="o4-mini"
LIMIT=200

# Add / Ingestion
echo "Running longbench_v2_ingestion.py..."
python -m evaluation.scripts.longbench_v2.longbench_v2_ingestion \
  --lib "$LIB" \
  --workers "$WORKERS" \
  --version-dir "$VERSION_DIR" \
  --mode "$ADD_MODE" \
  --async-mode "$ASYNC_MODE" \
  --limit "$LIMIT"

# #check
# echo "Running longbench_v2_check_files.py..."
# python -m evaluation.scripts.longbench_v2.longbench_v2_check_files \
#   --lib "$LIB" \
#   --version-dir "$VERSION_DIR" \

# # Search
# echo "Running longbench_v2_search.py..."
# python -m evaluation.scripts.longbench_v2.longbench_v2_search \
#  --lib "$LIB" \
#  --workers "$WORKERS" \
#  --version-dir "$VERSION_DIR" \
#  --top-k "$TOPK" \
#  --mode "$SEARCH_MODE" \
#  --limit "$LIMIT"

# Eval
# echo "Running longbench_v2_eval.py..."
# python -m evaluation.scripts.longbench_v2.longbench_v2_eval \
#  --lib "$LIB" \
#  --version-dir "$VERSION_DIR" \
#  --workers "$WORKERS" \
#  --chat-model "$CHAT_MODEL"

#echo "All scripts completed successfully!"
