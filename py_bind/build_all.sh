#!/usr/bin/env bash

# Exit on error
set -e

# Print commands before executing
set -x

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for python3
if ! command_exists python3; then
    echo "Python 3 is required but not installed. Aborting."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install build dependencies
python -m pip install \
    numpy>=1.21.0 \
    cython>=3.0.0 \
    setuptools>=61.0 \
    wheel \
    build \
    cmake>=3.15 \
    ninja

# Run the build script
echo "Running build script..."
python build.py

# Install the package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Run tests if pytest is available
if python -c "import pytest" >/dev/null 2>&1; then
    echo "Running tests..."
    pytest python/tests
else
    echo "pytest not found, skipping tests. Install with: pip install pytest"
fi

echo "Build completed successfully!"

# Deactivate virtual environment
deactivate