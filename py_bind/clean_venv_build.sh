#!/usr/bin/env bash

# Exit on error
set -e

# Print commands before executing
set -x

# Clean up any previous build artifacts
rm -rf build/
rm -rf *.egg-info/
rm -rf dist/
rm optv/*.c
rm -rf .venv*/
rm -rf liboptv/


# Copy liboptv headers for building
# mkdir -p liboptv/include
# cp -r ../liboptv/include/*.h liboptv/include/
# mkdir -p liboptv/src
# cp -r ../liboptv/src/*.c liboptv/src/

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Define Python versions to build for (3.11+)
PYTHON_VERSIONS=("3.11" "3.12" "3.13")

for py_version in "${PYTHON_VERSIONS[@]}"; do
    echo "Building for Python ${py_version}"
    
    # Create virtual environment with specific Python version
    uv venv --python="${py_version}" .venv-${py_version}
    source .venv-${py_version}/bin/activate

    # Install build dependencies
    uv pip install --upgrade pip
    uv pip install \
        scikit-build-core">=0.8.0" \
        cmake">=3.15" \
        ninja \
        cython">=3.0.0" \
        'numpy>=2.0.0,<3' \
        setuptools">=61.0.0" \
        pytest \
        build

    # Run build steps
    python setup.py prepare
    python setup.py build_ext --inplace
    python -m build --wheel --outdir dist/py${py_version}
    uv pip install dist/py${py_version}/*.whl --force-reinstall
    cd test && python -m pytest --verbose && cd ..


    # Deactivate virtual environment
    deactivate
done

# List all built wheels
echo "Built wheels:"
find dist/ -name "*.whl"
