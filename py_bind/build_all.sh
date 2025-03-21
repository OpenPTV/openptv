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

# Upgrade pip
python -m pip install --upgrade pip

# Install build dependencies
python -m pip install \
    numpy>=1.21.0 \
    cython>=3.0.0 \
    setuptools>=61.0 \
    wheel \
    pyyaml \
    pytest \
    build \
    cmake>=3.15 \
    ninja

# Build the C library first
echo "Building C library..."
cd ../liboptv
mkdir -p build
cd build
cmake ..
make
cd ../../py_bind

# Build and install the Python package
echo "Building and installing Python package..."
python setup.py build_ext --inplace
pip install -e .

# Run tests with correct PYTHONPATH
echo "Running tests..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd test
pytest
cd ..

echo "Build completed successfully!"

# Deactivate virtual environment
deactivate
