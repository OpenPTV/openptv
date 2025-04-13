#!/bin/bash
set -e

# Clean up any previous build artifacts
cd py_bind
rm -rf build dist *.egg-info optv/optv
find . -name "*.so" -o -name "*.c" | grep -v ".venv" | xargs rm -f 2>/dev/null || true

# Prepare the source files
python setup.py prepare

# Run cibuildwheel
cd ..
CIBW_TEST_REQUIRES="pytest" CIBW_TEST_COMMAND="cd {project}/py_bind/test && python -m pytest test_version.py" python -m cibuildwheel --output-dir wheelhouse py_bind/ --only "cp310-manylinux_x86_64 cp311-manylinux_x86_64"
