# -*- coding: utf-8 -*-
from __future__ import print_function
from distutils.core import setup
import setuptools
import os
import shutil

# We need to have Cython and Numpy installed.
# setup_requires does not play nice with setup_requires, so we need this hack. There are better hacks that
# work better (we need to check conda), but this may not be needed
try:
    from Cython.Distutils import build_ext
    from Cython.Distutils.extension import Extension
except ImportError:
    os.system('pip install cython')
    from Cython.Distutils import build_ext
    from Cython.Distutils.extension import Extension

try:
    import numpy as np
except ImportError:
    os.system('pip install numpy')
    import numpy as np

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

def copy_optv_sources():
    if not os.path.exists('../liboptv'):
        # This code is running from setup.py without the surrounding repository.
        # c-src should already be there
        return
    # Copy ../lipoptv/src aand .,/liboptv/inlucode to ./c-src/
    if os.path.exists('./c-src'):
        shutil.rmtree('./c-src')
    os.makedirs('./c-src')
    shutil.copytree('../liboptv/include', 'c-src/include/optv')
    shutil.copytree('../liboptv/src', 'c-src/src/optv')

inc_dirs = [np.get_include(), '.', './c-src/include', './c-src/include/optv']

def gather_c_sources():
    # Return all the C files in the liboptv/src directory.
    c_folder = './c-src/src'
    all_files = os.listdir(c_folder)
    c_files = [file for file in all_files if file.endswith('.c')]
    c_paths = [os.path.join(c_folder, c_file) for c_file in c_files]

    return c_paths

c_paths = None

def mk_ext(name, files, add_c_paths = True):
    global c_paths
    # Create a Cython extension, adding the C files
    if add_c_paths:
        if not c_paths:
            c_paths = gather_c_sources()
        files = files + c_paths
    return Extension(name, files, include_dirs=inc_dirs,
        pyrex_include_dirs=['.'])

copy_optv_sources()

ext_mods = [
    mk_ext("optv.tracking_framebuf", ["optv/tracking_framebuf.pyx"]),
    mk_ext("optv.parameters", ["optv/parameters.pyx"]),
    mk_ext("optv.calibration", ["optv/calibration.pyx"]),
    mk_ext("optv.transforms", ["optv/transforms.pyx"]),
    mk_ext("optv.imgcoord", ["optv/imgcoord.pyx"]),
    mk_ext("optv.image_processing", ["optv/image_processing.pyx"]),
    mk_ext("optv.correspondences", ["optv/correspondences.pyx"]),
    mk_ext("optv.segmentation", ["optv/segmentation.pyx"]),
    mk_ext("optv.epipolar", ["optv/epipolar.pyx"]),
    mk_ext("optv.tracker", ["optv/tracker.pyx"]),
    mk_ext("optv.orientation", ["optv/orientation.pyx"])
]

setup(
    name="optv",
    cmdclass = {'build_ext': build_ext},
    packages=['optv'],
    ext_modules = ext_mods,
    package_data = {'optv': ['*.pxd']},
    data_files = [('sources', ['c_src/src/*']), ('includes', ['c_src/include/*'])],
    version = '0.1.0',
    install_requires = [
        'numpy', 
        'pyyaml',
    ],
    setup_requires = ['numpy', 'cython']
)


