#!/bin/bash

# Common parameters for all scripts
LIB="supermemory"
WORKERS=10
TOPK=20
ADD_MODE="fine"
SEARCH_MODE="fine"
VERSION_DIR="test_1230"
ASYNC_MODE="sync"

#add
echo "Running mmlongbench_ingestion.py..."
python evaluation/scripts/mmlongbench/mmlongbench_ingestion.py --lib $LIB --workers $WORKERS --version-dir $VERSION_DIR --mode $ADD_MODE --async-mode $ASYNC_MODE
if [ $? -ne 0 ]; then
   echo "Error running mmlongbench_ingestion.py"
   exit 1
fi

#search
echo "Running mmlongbench_search.py..."
python evaluation/scripts/mmlongbench/mmlongbench_search.py --lib $LIB --workers $WORKERS --version-dir $VERSION_DIR --top-k $TOPK --mode $SEARCH_MODE
if [ $? -ne 0 ]; then
   echo "Error running mmlongbench_search.py"
   exit 1
fi

#eval
echo "Running mmlongbench_eval.py..."
python evaluation/scripts/mmlongbench/mmlongbench_eval.py --lib $LIB --version-dir $VERSION_DIR --workers $WORKERS
if [ $? -ne 0 ]; then
   echo "Error running mmlongbench_eval.py"
   exit 1
fi

echo "All scripts completed successfully!"
