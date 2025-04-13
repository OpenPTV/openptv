# OpenPTV

OpenPTV - framework for particle tracking velocimetry

## Project Structure

The OpenPTV project consists of two main components:

1. **liboptv** - The core C library implementing the particle tracking algorithms
2. **py_bind** - Python bindings for the C library

## Installation

### Prerequisites
- Python 3.10 or newer
- CMake 3.15 or newer
- C compiler (gcc, clang, or MSVC)
- pip

### Quick Install (Python Bindings)

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install build dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install build numpy cython
```

3. Build and install:
```bash
cd py_bind
python setup.py prepare
pip install -e .
```

4. Verify installation:
```bash
cd py_bind/test
python -m pytest
```

### Installing from PyPI

You can also install the pre-built wheels from PyPI:

```bash
pip install optv
```

## Developer Guide

### Running Tests

#### Python Tests

The Python tests are located in the `py_bind/test` directory and should be run from within that directory:

```bash
cd py_bind/test
python -m pytest                # Run all tests
python -m pytest test_version.py # Run a specific test file
python -m pytest -v              # Run with verbose output
```

#### C Library Tests

The C library tests use the Check framework and can be run in several ways:

1. Run all tests with debug output:
```bash
cd liboptv
mkdir -p build && cd build
cmake ../
CK_FORK=no CK_VERBOSITY=verbose ctest -V
```

2. Run a specific test (e.g., check_track) with debug output:
```bash
CK_FORK=no CK_VERBOSITY=verbose ctest -V -R check_track
```

3. Run a single test case with maximum debug information:
```bash
CTEST_OUTPUT_ON_FAILURE=1 CK_FORK=no CK_VERBOSITY=verbose CK_RUN_CASE="test_single_particle_track" ctest --output-on-failure -VV -R check_track
```

Environment variables explained:
- `CK_FORK=no`: Prevents Check from forking processes, ensuring all output is visible
- `CK_VERBOSITY=verbose`: Enables verbose output from Check framework
- `CK_RUN_CASE`: Specifies a single test case to run
- `CTEST_OUTPUT_ON_FAILURE=1`: Shows output for failed tests

### Building Wheels

You can build wheels for distribution using cibuildwheel:

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for the current platform
python -m cibuildwheel --output-dir wheelhouse py_bind/

# Build wheels for a specific Python version
python -m cibuildwheel --output-dir wheelhouse py_bind/ --only cp310-*
```

The project includes a GitHub Actions workflow that automatically builds wheels for multiple platforms when tags are pushed.

### Troubleshooting

If you encounter build issues, clean the build artifacts and try again:
```bash
cd py_bind
rm -rf build/
rm -rf *.egg-info/
rm -rf dist/
rm -rf optv/optv
find . -name "*.so" -o -name "*.c" | xargs rm -f
```

## Basic Usage

```python
from optv.tracking_framebuf import Target
from optv.tracker import Tracker
from optv.calibration import Calibration
```

## For C Library Users

```bash
cd liboptv
mkdir build && cd build
cmake ../
sudo make install
make verify
```

## Donations

Please consider donation to support our website and domain expenses and our developers during their job transitions.

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RK3FHXTCJDSWL)
