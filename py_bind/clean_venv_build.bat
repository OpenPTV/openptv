@echo off
setlocal enabledelayedexpansion

:: Exit on error
set "EXIT_ON_ERROR=1"

:: Enable command echoing
echo on

:: Clean up any previous build artifacts
rd /s /q build 2>nul
rd /s /q *.egg-info 2>nul
rd /s /q dist 2>nul
del /q optv\*.c 2>nul
rd /s /q .venv* 2>nul
rd /s /q liboptv 2>nul

:: Copy liboptv headers for building
:: mkdir liboptv\include 2>nul
:: xcopy /y /i ..\liboptv\include\*.h liboptv\include\
:: mkdir liboptv\src 2>nul
:: xcopy /y /i ..\liboptv\src\*.c liboptv\src\

:: Install uv if not already installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install-uv.ps1; .\install-uv.ps1"
    del install-uv.ps1
)

:: Define Python versions to build for
set "PYTHON_VERSIONS=3.10 3.11"

for %%v in (%PYTHON_VERSIONS%) do (
    echo Building for Python %%v
    
    :: Create virtual environment with specific Python version
    uv venv --python=%%v .venv-%%v
    call .venv-%%v\Scripts\activate.bat

    :: Install build dependencies
    uv pip install --upgrade pip
    uv pip install ^
        scikit-build-core">=0.8.0" ^
        cmake">=3.15" ^
        ninja ^
        cython">=3.0.0" ^
        numpy">=2.0.0" ^
        setuptools">=61.0.0" ^
        pytest ^
        build

    :: Run build steps
    python setup.py prepare
    python setup.py build_ext --inplace
    python -m build --wheel --outdir dist\py%%v
    uv pip install dist\py%%v\*.whl --force-reinstall
    cd test && python -m pytest --verbose && cd ..

    :: Deactivate virtual environment
    deactivate
)

:: List all built wheels
echo Built wheels:
dir /s /b dist\*.whl

endlocal