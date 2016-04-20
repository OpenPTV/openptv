# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import numpy as np
import os
inc_dirs = [np.get_include(), '.']

ext_mods = [
    Extension("optv.tracking_framebuf", ["optv/tracking_framebuf.pyx"], 
        libraries=['optv'], include_dirs=inc_dirs,
        pyrex_include_dirs=['.']),
    Extension("optv.parameters", ["optv/parameters.pyx"], 
        libraries=['optv'], include_dirs=inc_dirs,
        pyrex_include_dirs=['.']),
    Extension("optv.calibration", ["optv/calibration.pyx"], 
        libraries=['optv'], include_dirs=inc_dirs,
        pyrex_include_dirs=['.']),
    Extension("optv.imgcoord", ["optv/imgcoord.pyx"], 
        libraries=['optv'], include_dirs=inc_dirs,
        pyrex_include_dirs=['.']),
]

setup(
    name="optv",
    cmdclass = {'build_ext': build_ext},
    packages=['optv'],
    ext_modules = ext_mods,
    package_data = {'optv': ['*.pxd']}
)


