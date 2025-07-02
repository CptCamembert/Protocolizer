#!/bin/bash

set -e

echo "Starting Protocolizer (Real-time Audio Processing Client)..."

# Navigate to the protocolizer directory
cd "$(dirname "$0")"

# Check if diarization server is running
echo "Checking diarization server connection..."
if curl -s -X GET http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Diarization server is running and healthy"
else
    echo "❌ Error: Diarization server is not running!"
    echo "Please start the diarization server first:"
    echo "  cd ../server_side/diarization_server && ./docker/up.sh"
    exit 1
fi

# Activate virtual environment and run the protocolizer
echo "Starting real-time audio processing..."
echo "Press Ctrl+C to stop"
echo ""

/home/maximilian/diarization_clean/.venv/bin/python -m app.main_copy