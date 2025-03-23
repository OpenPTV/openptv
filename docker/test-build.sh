#!/usr/bin/env bash

# Exit on error
set -e

# Print commands
set -x

# Build the test Docker image
docker build -t optv-test-build -f docker/Dockerfile.test-build .

# Create a directory for the wheels
mkdir -p wheels

# Copy the wheels from the container
docker run --rm -v $(pwd)/wheels:/output optv-test-build sh -c "cp /wheels/* /output/"

# List the built wheels
echo "Built wheels:"
ls -l wheels/