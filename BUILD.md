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
