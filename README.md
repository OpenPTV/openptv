# OpenPTV

OpenPTV - framework for particle tracking velocimetry

## Installation

### Prerequisites
- Python 3.10 or newer
- CMake 3.15 or newer
- C compiler (gcc, clang, or MSVC)
- pip

### Quick Install

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
pytest test/
```

### Troubleshooting

If you encounter build issues, clean the build artifacts and try again:
```bash
rm -rf build/
rm -rf *.egg-info/
rm -rf dist/
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
