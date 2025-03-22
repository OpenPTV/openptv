#!/usr/bin/env bash

# Exit on error
set -e

# Print commands before executing
set -x

# Function to clean up build artifacts
cleanup_build_artifacts() {
    echo "Cleaning up build artifacts..."
    rm -rf build/
    rm -rf *.egg-info/
    rm -rf dist/
    rm -rf wheelhouse/
    rm -rf optv/*.c
    rm -rf venv/
    rm -rf liboptv/  # Clean up copied liboptv files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.so" -delete
}

# Clean up any previous build artifacts
cleanup_build_artifacts

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install build dependencies
python -m pip install --upgrade pip
python -m pip install \
    numpy>=1.21.0 \
    cython>=3.0.0 \
    setuptools>=61.0 \
    wheel \
    pyyaml \
    pytest

# Prepare source files (copies C sources and headers)
echo "Preparing source files..."
python setup.py prepare

# Build and install the Python package
echo "Building and installing Python package..."
python setup.py build_ext --inplace
pip install -e .

# Run tests
echo "Running tests..."
PYTHONPATH="${PYTHONPATH}:$(pwd)" pytest test/

echo "Build completed successfully!"

# Deactivate virtual environment
deactivate
