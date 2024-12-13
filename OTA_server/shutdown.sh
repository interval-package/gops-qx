#!/bin/bash

# PID file
PID_FILE="app.pid"

# Check if the PID file exists
if [[ ! -f "$PID_FILE" ]]; then
    echo "PID file not found. Is the application running?"
    exit 1
fi

# Read the PID from the file
PID=$(cat "$PID_FILE")

# Kill the process
if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID"
    echo "Application with PID $PID has been shut down."
    rm "$PID_FILE"  # Remove the PID file after shutdown
else
    echo "No process found with PID $PID."
    rm "$PID_FILE"  # Remove the PID file if process is not found
fi
