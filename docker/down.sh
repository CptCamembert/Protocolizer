#!/bin/bash

echo "Stopping protocolizer container..."

# Check if the container exists
if docker ps -a --format '{{.Names}}' | grep -q '^protocolizer$'; then
  # Stop and remove the container
  docker stop protocolizer
  docker rm protocolizer
  echo "Protocolizer container stopped and removed."
else
  echo "Protocolizer container is not running."
fi