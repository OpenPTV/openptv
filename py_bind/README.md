# OpenPTV Python Bindings

Python bindings for the OpenPTV library.

## Installation

Simple installation via pip:

```bash
pip install openptv
```

## Building from Source

1. Install dependencies:
```bash
pip install numpy cython setuptools
```

2. Build the package:
```bash
python setup.py prepare
python setup.py build
python setup.py install
```

## Testing

Run the test suite:
```bash
pytest test/
```

## License

LGPL-3.0-or-later