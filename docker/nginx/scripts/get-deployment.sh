#!/bin/bash
# Get the currently active deployment color

# Read from environment file
if [ -f /etc/environment ]; then
    source /etc/environment
    echo $ACTIVE_DEPLOYMENT
else
    # Default to blue if not set
    echo "blue"
fi