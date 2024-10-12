#!/bin/bash

# Get the directory of the script
script_dir=$(dirname "$(realpath "$0")")

# Get the current working directory
current_dir=$(pwd)

# Check if the current directory is the same as the script directory
if [ "$script_dir" != "$current_dir" ]; then
    echo "Error: Current working directory is not the script directory."
    echo "Please change to the directory: $script_dir"
    exit 1
fi

# Define the PID file
pid_file="localqxsim_service.pid"

# Check if the PID file exists
if [ ! -f "$pid_file" ]; then
    echo "Error: PID file '$pid_file' not found."
    exit 1
fi

# Read the PID from the file
pid=$(cat "$pid_file")

# Check if the PID is a valid number
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid PID in file."
    exit 1
fi

# Check if the process with the PID exists
if ps -p "$pid" > /dev/null 2>&1; then
    # Kill the process
    kill "$pid"
    echo "Process $pid killed."
else
    echo "No process found with PID $pid."
fi
