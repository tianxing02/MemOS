#!/bin/bash

# Get the directory where this script is located (MemOS/evaluation)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# The project root is the parent directory of 'evaluation' (MemOS)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Navigate to the project root so that scripts running from root work correctly
echo "Changing directory to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Failed to change directory to $PROJECT_ROOT"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p evaluation/logs

# Define the log file with timestamp
LOG_FILE="evaluation/logs/weekly_eval_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================" | tee -a "$LOG_FILE"
echo "Starting weekly evaluation tasks at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Function to run a script and log its output
run_script() {
    script_path=$1
    echo "" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running script: $script_path" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"

    if [ -f "$script_path" ]; then
        # Run the script using bash
        # We assume the script is executable or runnable via bash
        bash "$script_path" >> "$LOG_FILE" 2>&1
        status=$?

        if [ $status -eq 0 ]; then
            echo "SUCCESS: $script_path completed successfully." | tee -a "$LOG_FILE"
        else
            echo "FAILURE: $script_path failed with exit code $status." | tee -a "$LOG_FILE"
        fi
    else
        echo "ERROR: Script file not found: $script_path" | tee -a "$LOG_FILE"
    fi
}

# List of scripts to run
# Paths are relative to the project root
run_script "evaluation/scripts/run_lme_eval.sh"
run_script "evaluation/scripts/run_locomo_eval.sh"
run_script "evaluation/scripts/run_pm_eval.sh"
run_script "evaluation/scripts/run_prefeval_eval.sh"

echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "All tasks finished at $(date)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
