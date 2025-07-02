#!/bin/bash

set -e

echo "Starting protocolizer..."

# Check if diarization server is running
echo "Checking diarization server connection..."
if curl -s -X GET http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Diarization server is running and healthy"
else
    echo "❌ Error: Diarization server is not running!"
    echo "Please start the diarization server first:"
    echo "  cd ../../server_side/diarization_server && ./docker/up.sh"
    exit 1
fi

# Run the Docker container with audio device access
# --device /dev/snd: Access to audio devices
# --privileged: Required for audio access in some systems
# -v /tmp/.X11-unix:/tmp/.X11-unix: X11 forwarding if needed
# -e DISPLAY: Display environment for GUI apps
# --network host: Use host networking to access diarization server on localhost
docker run -d \
  --device /dev/snd \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY="$DISPLAY" \
  --network host \
  --name protocolizer \
  protocolizer

echo "Protocolizer started successfully!"
echo "The protocolizer is now capturing audio and sending it to the diarization server."
echo "Check logs with: docker logs protocolizer"
echo "Stop with: ./docker/down.sh"