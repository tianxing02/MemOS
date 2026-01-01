#!/bin/bash

# Common parameters for all scripts
LIB="supermemory"
WORKERS=20
TOPK=20
ADD_MODE="fine"
SEARCH_MODE="fine"
VERSION_DIR="test_1231_1"
ASYNC_MODE="sync"
CHAT_MODEL="gpt-4o-mini"

#add
echo "Running hotpot_ingestion.py..."
python evaluation/scripts/hotpot/hotpot_ingestion.py --lib $LIB --workers $WORKERS --version-dir $VERSION_DIR --mode $ADD_MODE --async-mode $ASYNC_MODE --limit 20
if [ $? -ne 0 ]; then
   echo "Error running hotpot_ingestion.py"
   exit 1
fi

#search
echo "Running hotpot_search.py..."
python evaluation/scripts/hotpot/hotpot_search.py --lib $LIB --workers $WORKERS --version-dir $VERSION_DIR --top-k $TOPK --mode $SEARCH_MODE --limit 20
if [ $? -ne 0 ]; then
   echo "Error running hotpot_search.py"
   exit 1
fi

#eval
echo "Running hotpot_eval.py..."
python evaluation/scripts/hotpot/hotpot_eval.py --lib $LIB --version-dir $VERSION_DIR --workers $WORKERS --chat-model $CHAT_MODEL
if [ $? -ne 0 ]; then
   echo "Error running hotpot_eval.py"
   exit 1
fi

echo "All scripts completed successfully!"
