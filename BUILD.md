# Building OpenPTV Python Package

## Prerequisites

- Python 3.10 or newer
- CMake 3.15 or newer
- C compiler (gcc, clang, or MSVC)
- pip

## Build Steps

1. Install build dependencies:
```bash
pip install numpy cython
```

2. Change to the py_bind directory and prepare the sources:
```bash
cd py_bind
python setup.py prepare
```

3. Install the package:
```bash
pip install -e .
```

## Verifying the Installation

### 1. Check Python Package Installation

```bash
# Check if the package is installed
pip list | grep optv

# Verify import works in Python
python -c "import optv; print(optv.__version__)"

# Check all core modules import correctly
python -c """
import optv
from optv import tracking_framebuf
from optv import parameters
from optv import calibration
from optv import correspondences
from optv import image_processing
from optv import orientation
from optv import segmentation
from optv import tracking_framebuf
print('All modules imported successfully')
"""
```

### 2. Check Library Files

```bash
# On Linux/MacOS
find $(python -c "import optv; print(optv.__path__[0])") -name "*.so"

# On Windows
dir /s $(python -c "import optv; print(optv.__path__[0])") *.pyd

# Check liboptv installation
# Linux
ldconfig -p | grep liboptv
# MacOS
ls /usr/local/lib/liboptv*
# Windows
dir /s "C:\Program Files\OpenPTV\lib\liboptv*"
```

### 3. Verify Compiler Setup

```bash
# Check C compiler
gcc --version  # or clang --version on MacOS
# On Windows
cl.exe  # Should show MSVC version

# Check CMake
cmake --version

# Verify Python development headers
python -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
```

### 4. Run Test Suite

```bash
# Run unit tests
cd test
pytest -v

# Run specific test categories
pytest test_tracking.py -v
pytest test_calibration.py -v
pytest test_parameters.py -v
```

### 5. Functional Tests

```python
# Save as test_installation.py
import numpy as np
from optv import tracking_framebuf
from optv import parameters
from optv import calibration

def test_basic_functionality():
    # Test reading parameters
    control_params = parameters.ControlParams(n_img=4)
    assert control_params.n_img == 4

    # Test calibration
    cal = calibration.Calibration()
    assert cal is not None

    # Test array handling
    arr = np.array([[1., 2.], [3., 4.]])
    # Your specific array operation test here

if __name__ == "__main__":
    test_basic_functionality()
    print("Basic functionality test passed")
```

### 6. Check Build Configuration

```bash
# Show build configuration
python -c """
import optv
import sys
import platform

print(f'Python version: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'OpenPTV version: {optv.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Compiler used: {platform.python_compiler()}')
"""
```

## Common Issues and Troubleshooting

### Compiler Issues
- **Linux**: Install build essentials: `sudo apt-get install build-essential`
- **MacOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Windows**: Install Visual Studio Build Tools with C++ workload

### Python Issues
- Ensure Python headers are installed: `python3-dev` on Linux
- Check Python architecture matches compiler (32/64 bit)
- Verify virtual environment is activated if using one

### Library Issues
- **Linux**: Run `sudo ldconfig` after installation
- **MacOS**: Check `DYLD_LIBRARY_PATH`
- **Windows**: Verify PATH includes library directory

### Build Errors
- Clean build directories before rebuilding:
```bash
rm -rf build/
rm -rf *.egg-info/
rm -rf dist/
```
- Rebuild with verbose output:
```bash
pip install -v -e .
```

If you encounter any issues, please check the error messages and consult the troubleshooting section above. For further assistance, create an issue on our GitHub repository with the full error output and your system information.
