#!/bin/bash

# Common parameters for all scripts
LIB="supermemory"
WORKERS=5
TOPK=20
ADD_MODE="fine"
SEARCH_MODE="fine"
VERSION_DIR="test_1231"
ASYNC_MODE="sync"
CHAT_MODEL="gpt-4o-mini"

#add
#echo "Running longbench_v2_ingestion.py..."
#python evaluation/scripts/longbench_v2/longbench_v2_ingestion.py --lib $LIB --workers $WORKERS --version-dir $VERSION_DIR --mode $ADD_MODE --async-mode $ASYNC_MODE
#if [ $? -ne 0 ]; then
#   echo "Error running longbench_v2_ingestion.py"
#   exit 1
#fi

#search
echo "Running longbench_v2_search.py..."
python evaluation/scripts/longbench_v2/longbench_v2_search.py --lib $LIB --workers $WORKERS --version-dir $VERSION_DIR --top-k $TOPK --mode $SEARCH_MODE --limit 30
if [ $? -ne 0 ]; then
   echo "Error running longbench_v2_search.py"
   exit 1
fi

#eval
echo "Running longbench_v2_eval.py..."
python evaluation/scripts/longbench_v2/longbench_v2_eval.py --lib $LIB --version-dir $VERSION_DIR --workers $WORKERS --chat-model $CHAT_MODEL
if [ $? -ne 0 ]; then
   echo "Error running longbench_v2_eval.py"
   exit 1
fi

echo "All scripts completed successfully!"
