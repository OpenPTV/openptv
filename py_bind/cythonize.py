#!/usr/bin/env python
import os
import glob
from Cython.Build import cythonize
import numpy

def main():
    # Get all .pyx files
    pyx_files = glob.glob('optv/*.pyx')
    
    # Configure Cython compilation
    extensions = cythonize(
        pyx_files,
        compiler_directives={
            'language_level': '3',
            'binding': True,
            'embedsignature': True,
        },
        include_path=[
            numpy.get_include(),
            '.',
            'optv',
            './liboptv/include',
            './liboptv/include/optv',
            './liboptv/src'
        ],
        force=True  # Force recompilation
    )

if __name__ == '__main__':
    main()
