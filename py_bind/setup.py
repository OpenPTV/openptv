# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import numpy as np
import os
inc_dirs = [np.get_include(), '.']

def mk_ext(name, files):
    return Extension(name, files, libraries=['optv'], include_dirs=inc_dirs,
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

setup(
    name="optv",
    cmdclass = {'build_ext': build_ext},
    packages=['optv'],
    ext_modules = ext_mods,
    package_data = {'optv': ['*.pxd']}
)


