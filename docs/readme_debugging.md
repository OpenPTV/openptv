# OpenPTV liboptv Developer README

## Overview

This library (`liboptv`) is part of the OpenPTV project and provides core algorithms for particle tracking velocimetry, including 2D and 3D tracking routines. The codebase is C99, uses CMake for building, and includes a comprehensive suite of unit tests using the [Check](https://libcheck.github.io/check/) framework.

This README is for developers who want to **debug**, **extend**, or **test** the library, especially using Visual Studio Code (VS Code) or other modern IDEs.

---

## Prerequisites

- **Linux** (tested on Ubuntu)
- **CMake** (>=3.10)
- **GCC** (with gdb for debugging)
- **Check** unit testing framework (`libcheck-dev`)
- **VS Code** (recommended, with C/C++ extension)
- **Electric Fence** (`electric-fence`, optional, for memory debugging)

Install dependencies (Ubuntu example):

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libcheck-dev electric-fence gdb
```

---

## Building the Library

```bash
cd /path/to/openptv/liboptv
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

- The default build type is **Debug** (with debug symbols, no optimization).
- The shared library will be built as `liboptv.so` in `build/src/`.

---

## Running Tests

To run all tests:

```bash
make verify
```

Or, using CTest:

```bash
cd build
ctest
```

To run a specific test (e.g., only 3D tracking):

```bash
ctest -R track3d
```

To see output on failure:

```bash
CTEST_OUTPUT_ON_FAILURE=1 ctest -V -R track3d
```

---

## Debugging with VS Code

1. **Open the project root in VS Code:**

    ```bash
    code /path/to/openptv/liboptv
    ```

2. **Build in Debug mode** (see above).

3. **Set up `.vscode/launch.json`:**

    Example for debugging the 3D tracking test:

    ```json
    {
      "version": "0.2.0",
      "configurations": [
        {
          "name": "Debug check_track3d",
          "type": "cppdbg",
          "request": "launch",
          "program": "${workspaceFolder}/build/tests/check_track3d",
          "args": [],
          "stopAtEntry": false,
          "cwd": "${workspaceFolder}/build/tests",
          "environment": [],
          "externalConsole": false,
          "MIMode": "gdb",
          "setupCommands": [
            {
              "description": "Enable pretty-printing for gdb",
              "text": "-enable-pretty-printing",
              "ignoreFailures": true
            }
          ]
        }
      ]
    }
    ```

4. **Set breakpoints** in any source file (e.g., `src/track3d.c`).

5. **Start debugging** from the Run & Debug panel.

---

## Tips for Debugging

- **Step into library code:** Ensure you build with `-g -O0` (Debug mode, no optimization).
- **Flush output:** Add `fflush(stdout);` after `printf` to see output immediately during tests.
- **Run a single test:** Use `ctest -R testname` or pass arguments to your test executable.
- **Check source paths:** Open the original source files in VS Code for breakpoints to work.

---

## Directory Structure

- `src/` — Core library source files
- `include/` — Public headers
- `tests/` — Unit and integration tests (Check framework)
- `build/` — Build directory (created by you)
- `.vscode/` — VS Code configuration (optional)

---

## Common Issues

- **Cannot step into src code:**  
  Make sure you built with debug symbols and no optimization. Clean and rebuild if needed.
- **Linker errors for efence:**  
  Install `electric-fence` or comment out its usage in `CMakeLists.txt`.
- **CTest output is buffered:**  
  Use `fflush(stdout);` or run with `CTEST_OUTPUT_ON_FAILURE=1`.

---

## Contributing

- Follow the code style of existing files.
- Add tests for new features or bugfixes.
- Document your changes.

---

## Further Help

If you have questions or issues, open an issue on the OpenPTV GitHub or contact the maintainers.

---