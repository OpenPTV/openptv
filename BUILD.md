# Building OpenPTV Python Package

## Prerequisites

- Python 3.10 or newer
- CMake 3.15 or newer
- C compiler (gcc, clang, or MSVC)
- pip

## Build Steps

1. Install build dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install build numpy cython
```

2. Build the package:
```bash
# From the root directory
python -m build
```

3. Install the built package:
```bash
pip install dist/*.whl
```

## Creating Binary Wheels

### Local Binary Wheels

To create platform-specific binary wheels:

1. Navigate to the py_bind directory:
```bash
cd py_bind
```

2. Clean previous build artifacts:
```bash
rm -rf build/ dist/ *.egg-info/
rm -rf optv/*.c  # Remove Cython-generated files
```

3. Prepare the source files:
```bash
python setup.py prepare
```

4. Build the wheel:
```bash
python -m pip wheel . -w dist/
```

The wheel will be created in the `dist/` directory with a platform-specific name like:
- Linux: `openptv-0.3.0-cp311-cp311-linux_x86_64.whl`
- Windows: `openptv-0.3.0-cp311-cp311-win_amd64.whl`
- macOS: `openptv-0.3.0-cp311-cp311-macosx_10_9_x86_64.whl`

### Building Wheels for Multiple Platforms

For building wheels for multiple platforms and Python versions, we use `cibuildwheel`:

1. Install cibuildwheel:
```bash
pip install cibuildwheel
```

2. Build wheels for all supported platforms:
```bash
# Linux/macOS
python -m cibuildwheel --platform linux --arch x86_64
# or
python -m cibuildwheel --platform macos --arch x86_64
# or on Windows
python -m cibuildwheel --platform windows --arch AMD64
```

The wheels will be available in the `wheelhouse/` directory.

## Development Installation

For development, you can install the package in editable mode:

```bash
cd py_bind
python setup.py prepare
pip install -e .
```

## Running Tests

```bash
pytest python/tests
```

## Troubleshooting

### Build Issues
- Clean build artifacts before rebuilding:
```bash
rm -rf build/
rm -rf *.egg-info/
rm -rf dist/
```

- Rebuild with verbose output:
```bash
python -m build -v
```

### Python Issues
- Ensure Python headers are installed: `python3-dev` on Linux
- Check Python architecture matches compiler (32/64 bit)
- Verify virtual environment is activated if using one

### Library Issues
- **Linux**: Run `sudo ldconfig` after installation
- **MacOS**: Check `DYLD_LIBRARY_PATH`
- **Windows**: Verify PATH includes library directory
