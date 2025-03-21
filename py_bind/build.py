import os
import shutil
import glob
from pathlib import Path
from typing import List

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import numpy
        import Cython
    except ImportError as e:
        print("Error: Missing required dependencies.")
        print("Please install required packages:")
        print("pip install numpy cython")
        raise SystemExit(1)

def prepare_sources() -> None:
    """Prepare the C sources by copying them from liboptv and converting pyx to C"""
    # Check dependencies first
    check_dependencies()
    
    # Create necessary directories
    os.makedirs('liboptv/include', exist_ok=True)
    os.makedirs('liboptv/src', exist_ok=True)
    
    # Copy liboptv sources
    for c_file in glob.glob('../liboptv/src/*.c'):
        print(f"Copying source: {c_file}")
        shutil.copy(c_file, 'liboptv/src/')
    
    # Copy liboptv headers
    for h_file in glob.glob('../liboptv/include/*.h'):
        print(f"Copying header: {h_file}")
        dest = os.path.join('liboptv/include', os.path.basename(h_file))
        shutil.copy(h_file, dest)
        
        # Also copy headers to the root liboptv directory for compatibility
        dest = os.path.join('liboptv', os.path.basename(h_file))
        shutil.copy(h_file, dest)
    
    # Convert pyx to C
    from Cython.Build import cythonize
    cythonize(['optv/*.pyx'], compiler_directives={'language_level': '3'})

if __name__ == "__main__":
    prepare_sources()
