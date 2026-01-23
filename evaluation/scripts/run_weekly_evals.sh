#!/bin/bash
set -e

# Get the directory where this script is located (MemOS/evaluation/scripts)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Project root (MemOS)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Changing directory to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Failed to change directory"; exit 1; }

# Base logs directory
LOG_BASE="evaluation/logs"

# Timestamp (shared by all logs in this run)
TS="$(date +%Y%m%d_%H%M%S)"

# Run-specific log directory
RUN_LOG_DIR="${LOG_BASE}/${TS}"

# Master log file
MASTER_LOG="${RUN_LOG_DIR}/weekly_eval_${TS}.log"

# 🔴 IMPORTANT: create log directories BEFORE using tee
mkdir -p "$RUN_LOG_DIR"

# Environment variables

export LIB="memos-api-online"
export VERSION="weekly_eval_$(date +%Y%m%d)"

# Master log header
echo "==================================================" | tee -a "$MASTER_LOG"
echo "Starting weekly evaluation tasks at $(date)" | tee -a "$MASTER_LOG"
echo "Project root: $PROJECT_ROOT" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "LIB: $LIB" | tee -a "$MASTER_LOG"
echo "VERSION: $VERSION" | tee -a "$MASTER_LOG"
echo "==================================================" | tee -a "$MASTER_LOG"

# Dataset check
echo "Running dataset check..." | tee -a "$MASTER_LOG"

python evaluation/scripts/download_datasets.py >> "$MASTER_LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset check failed." | tee -a "$MASTER_LOG"
    exit 1
fi

# Function to run a script with its own log
run_script() {
    local script_path="$1"
    local script_name
    script_name="$(basename "$script_path" .sh)"

    local script_log="${RUN_LOG_DIR}/${script_name}_${TS}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "--------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "Running script: $script_path" | tee -a "$MASTER_LOG"
    echo "Script log: $script_log" | tee -a "$MASTER_LOG"
    echo "Start time: $(date)" | tee -a "$MASTER_LOG"
    echo "--------------------------------------------------" | tee -a "$MASTER_LOG"

    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script file not found: $script_path" | tee -a "$MASTER_LOG"
        return 1
    fi

    echo "[$(date)] START $script_path" >> "$script_log"

    bash "$script_path" >> "$script_log" 2>&1
    local status=$?

    if [ $status -eq 0 ]; then
        echo "[$(date)] SUCCESS $script_path" >> "$script_log"
        echo "SUCCESS: $script_path completed successfully." | tee -a "$MASTER_LOG"
    else
        echo "[$(date)] FAILURE $script_path (exit code $status)" >> "$script_log"
        echo "FAILURE: $script_path failed with exit code $status." | tee -a "$MASTER_LOG"
        return $status
    fi
}

# run_script "evaluation/scripts/run_lme_eval.sh"
# run_script "evaluation/scripts/run_locomo_eval.sh"
# run_script "evaluation/scripts/run_pm_eval.sh"
run_script "evaluation/scripts/run_prefeval_eval.sh"

# Finish
echo "" | tee -a "$MASTER_LOG"
echo "==================================================" | tee -a "$MASTER_LOG"
echo "All tasks finished at $(date)" | tee -a "$MASTER_LOG"
echo "==================================================" | tee -a "$MASTER_LOG"
