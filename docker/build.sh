#!/bin/bash

set -e

echo "Building protocolizer Docker image..."

# Navigate to the parent directory where Dockerfile is located
cd "$(dirname "$0")/.."

# Build the Docker image
docker build -t protocolizer .

echo "Docker image 'protocolizer' built successfully!"