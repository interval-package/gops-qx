#!/bin/bash

# Configuration file
CONFIG_FILE="config.txt"
PID_FILE="app.pid"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/app.log"
PYTHON_SCRIPT="your_python_script.py"  # Replace with your actual script name

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Read kwargs from the config file
ARGS=()
while IFS=': ' read -r key value; do
    ARGS+=("--$key" "$value")
done < "$CONFIG_FILE"

# Run the Python script in the background and save the PID
python "$PYTHON_SCRIPT" "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Application started with PID $(cat "$PID_FILE")."
