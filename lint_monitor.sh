#!/usr/bin/env bash

LOG_FILE="pylint_monitor.log"

while true; do
    # Run pylint and capture output
    OUTPUT=$(pylint evoprompt/**py)
    
    # Get last line of output
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)
    
    # Get current datetime
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Append to log file and echo to stdout
    echo "$TIMESTAMP - $LAST_LINE" | tee -a "$LOG_FILE"
    
    # Wait 60 seconds
    sleep 60
done
