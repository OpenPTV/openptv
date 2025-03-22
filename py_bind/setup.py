from setuptools import setup, Extension, Command
import os
import glob
import shutil
import sys
import numpy as np

class PrepareCommand(Command):
    """Prepare the C sources by copying them from liboptv and converting pyx to C"""
    description = "Prepare the C sources and convert pyx files to C"
    user_options = []
    
    def initialize_options(self):
        pass
        
    def finalize_options(self):
        pass
        
    def run(self):
        """Prepare the C sources by copying them from liboptv and converting pyx to C"""
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
        
        # Convert pyx to C
        from Cython.Build import cythonize
        cythonize(['optv/*.pyx'], compiler_directives={'language_level': '3'})

# Common include directories
include_dirs = ['liboptv/include', np.get_include()]

# Common compiler and linker args
extra_compile_args = ['-Wno-cpp', '-Wno-unused-function'] if not sys.platform.startswith('win') else ['/W4']
extra_link_args = ['-Wl,-rpath,$ORIGIN'] if not sys.platform.startswith('win') else []

# Define extension modules by finding all pyx files
pyx_files = glob.glob('optv/*.pyx')
ext_modules = [
    Extension(
        f"optv.{os.path.splitext(os.path.basename(pyx_file))[0]}", 
        sources=[pyx_file],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    for pyx_file in pyx_files
]

setup(
    name='optv',
    packages=['optv'],
    package_dir={'optv': 'optv'},
    ext_modules=ext_modules,
    cmdclass={
        'prepare': PrepareCommand,
    },
    include_package_data=True,
)
