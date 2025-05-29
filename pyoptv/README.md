# PyOptv - Pure Python OpenPTV

PyOptv is a pure Python implementation of OpenPTV (Open Source Particle Tracking Velocimetry), providing tools for 3D particle tracking velocimetry analysis.

## Features

- Camera calibration and orientation
- Image processing and particle detection
- 3D correspondence and tracking
- Ray tracing and stereoscopic reconstruction
- Pure Python implementation (no compiled dependencies)
- Optional Numba acceleration for performance-critical operations

## Installation

### From source

```bash
cd pyoptv
pip install -e .
```

### Development installation

```bash
cd pyoptv
pip install -e ".[dev]"
```

This will install the package in development mode with all development dependencies including testing tools.

## Usage

```python
import pyoptv
from pyoptv.calibration import Calibration
from pyoptv.tracking_frame_buf import FrameBuf

# Create a calibration object
cal = Calibration()

# Set camera parameters
cal.set_pos([100.0, 200.0, 300.0])
cal.set_angles([0.1, 0.2, 0.3])
```

## Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=pyoptv
```

## Dependencies

- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

### Optional dependencies

- Numba >= 0.56.0 (for performance acceleration)

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Contributing

Please see the main OpenPTV repository for contribution guidelines.
