# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import numpy as np
import os
import shutil

# The python bindings have been redone, so they do not require the liboptv.so to be installed.
# We do the following:
#
# Copy the headerfiles from ../liboptv/include to include/optv . This is because the Python code expects header
# files under the optv directory, like is installed in /usr/local/include .
# 
# Find all the C source files from liboptv/src and add them to each extension, so they are compiled with the extension.
# This may seem expensive, as the files are added to each compiled extension (as if using a static liboptv library)
# in the future we may unite Cython modules into one extension (really not straightforward) and save the extra space.
# 
# Tell Cython to look for header files in include/ (for the Cythong code) and include/optv (for the C code) 

def copy_optv_headers():
    # Copy ../liboptv/include - the C sources include directory, to our internal include/optv directory
    if not os.path.exists('./include'):
        os.makedirs('./include')
    shutil.rmtree('./include/optv')
    shutil.copytree('../liboptv/include', 'include/optv')

inc_dirs = [np.get_include(), '.', './include', './include/optv']

def gather_c_sources():
    # Return all the C files in the liboptv/src directory.
    c_folder = '../liboptv/src'
    all_files = os.listdir(c_folder)
    c_files = [file for file in all_files if file.endswith('.c')]
    c_paths = [os.path.join(c_folder, c_file) for c_file in c_files]

    return c_paths

c_paths = gather_c_sources()

def mk_ext(name, files, add_c_paths = True):
    # Create a Cython extension, adding the C files
    if add_c_paths:
        files = files + c_paths
    return Extension(name, files, include_dirs=inc_dirs,
        pyrex_include_dirs=['.'])

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

copy_optv_headers()
setup(
    name="optv",
    cmdclass = {'build_ext': build_ext},
    packages=['optv'],
    ext_modules = ext_mods,
    package_data = {'optv': ['*.pxd']},
    version = '0.1.0',
)


