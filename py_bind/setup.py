# -*- coding: utf-8 -*-

from distutils.core import setup
import setuptools
import os
import shutil
import sys
import glob
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import importlib
import numpy


class PrepareCommand(setuptools.Command):
    """Prepare the C sources by copying them from liboptv and converting pyx to C"""
    
    description = "Prepare C sources and Cython files"
    user_options = []
    
    def initialize_options(self): pass
    def finalize_options(self): pass
    
    def run(self):
        # Create necessary directories
        if not os.path.exists('liboptv'):
            os.mkdir('liboptv')
        if not os.path.exists('liboptv/include'):
            os.makedirs('liboptv/include', exist_ok=True)
        if not os.path.exists('liboptv/src'):
            os.makedirs('liboptv/src', exist_ok=True)        
        
        # Copy liboptv sources
        for c_file in glob.glob('../liboptv/src/*.c'):
            shutil.copy(c_file, 'liboptv/src/')
        
        # Copy liboptv headers
        for h_file in glob.glob('../liboptv/include/*.h'):
            shutil.copy(h_file, 'liboptv/include/')
        
        # Convert pyx to C
        from Cython.Build import cythonize
        cythonize(['optv/*.pyx'])


class BuildExt(build_ext):  # Remove unnecessary (object) inheritance
    def run(self):
        if not os.path.exists('./liboptv') or not glob.glob('./optv/*.c'):
            print('You must run setup.py prepare before building the extension', file=sys.stderr)
            raise Exception('You must run setup.py prepare before building the extension')
        self.add_include_dirs()
        super().run()  # Simplified super() call for Python 3

    @staticmethod
    def get_numpy_include_dir():
        # Simplified numpy include directory detection for Python 3
        import builtins
        if hasattr(builtins, '__NUMPY_SETUP__'):
            del builtins.__NUMPY_SETUP__
        import numpy
        importlib.reload(numpy)

        return numpy.get_include()

    def add_include_dirs(self):
        np_include_dir = self.get_numpy_include_dir()
        include_dirs = [np_include_dir, '.', './liboptv/include', './liboptv/include/optv']

        for extension in self.extensions:
            extension.include_dirs = include_dirs

# The python bindings have been redone, so they do not require the liboptv.so to be installed.
# We do the following:
#
# Copy all the C sources, to c-src, as setup.py only packs files under setup.py in the ZIP file
# 
# Find all the C source files from liboptv/src and add them to each extension, so they are compiled with the extension.
# This may seem expensive, as the files are added to each compiled extension (as if using a static liboptv library)
# in the future we may unite Cython modules into one extension (really not straightforward) and save the extra space.
# 
# Tell Cython to look for header files in c-src/include/ (for the Cython code) and c-src/include/optv (for the C code) 

def get_liboptv_sources():
    return glob.glob('./liboptv/src/*.c')


def mk_ext(name, files):
    extra_compile_args = []
    if not sys.platform.startswith('win'):
        extra_compile_args.append('-Wno-cpp')
    else:
        extra_compile_args.append('/W4')  # Add Windows warning level
        
    return Extension(
        name,
        files + get_liboptv_sources(),
        extra_compile_args=extra_compile_args
    )


ext_mods = [
    mk_ext("optv.tracking_framebuf", ["optv/tracking_framebuf.c"]),
    mk_ext("optv.parameters", ["optv/parameters.c"]),
    mk_ext("optv.calibration", ["optv/calibration.c"]),
    mk_ext("optv.transforms", ["optv/transforms.c"]),
    mk_ext("optv.imgcoord", ["optv/imgcoord.c"]),
    mk_ext("optv.image_processing", ["optv/image_processing.c"]),
    mk_ext("optv.correspondences", ["optv/correspondences.c"]),
    mk_ext("optv.segmentation", ["optv/segmentation.c"]),
    mk_ext("optv.epipolar", ["optv/epipolar.c"]),
    mk_ext("optv.tracker", ["optv/tracker.c"]),
    mk_ext("optv.orientation", ["optv/orientation.c"])
]

setup(
    name="optv",
    cmdclass={
        'build_ext': BuildExt,
        'prepare': PrepareCommand,
    },
    packages=['optv'],
    ext_modules=ext_mods,
    include_package_data=True,
    data_files=[
        ('liboptv', glob.glob('liboptv/src/*.c') + glob.glob('liboptv/include/optv/*.h'))
    ],
    package_data={
        'optv': ['*.pxd', '*.c', '*.h'],
    },
    version='0.3.0',
    install_requires=[
        'numpy',
        'cython',
        'pyyaml',
        'matplotlib'
    ],
    setup_requires=['numpy', 'cython'],
    python_requires='>=3.10',
)
