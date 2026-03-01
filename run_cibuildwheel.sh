#!/bin/bash
set -e

# Clean up any previous build artifacts
cd py_bind
rm -rf build dist *.egg-info optv/optv
find . -name "*.so" -o -name "*.c" | grep -v ".venv" | xargs rm -f 2>/dev/null || true

# Install dependencies (host Python must be 3.11+ for cibuildwheel 3.x)
python -m pip install --upgrade pip
python -m pip install numpy>=2.0.0 cython>=3.0.0

# Prepare the source files
python setup.py prepare

# Run cibuildwheel for Python 3.11, 3.12, 3.13
cd ..
CIBW_BUILD="cp311-* cp312-* cp313-*" \
CIBW_SKIP="*musllinux*" \
CIBW_TEST_REQUIRES="pytest" \
CIBW_TEST_COMMAND="cd {project}/py_bind/test && python -m pytest test_version.py" \
python -m cibuildwheel --output-dir wheelhouse py_bind/
