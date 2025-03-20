# Building OpenPTV Python Package

## Build Steps

1. Change to the py_bind directory:
```bash
cd py_bind
```

2. Run the prepare command to copy liboptv sources and convert pyx files to C:
```bash
python setup.py prepare
```

3. Install the package:
```bash
pip install .
```

## Testing the Installation

After installation, run the tests:
```bash
cd test
pytest
```

## Common Issues

- If you get an error during build, make sure you ran the `prepare` command first
- Ensure you have numpy<1.24 and cython<3 installed before building