# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import os

def get_liboptv_sources():
    """Get all required C source files"""
    base_sources = [
        os.path.join('liboptv', 'src', src) for src in [
            'parameters.c',      
            'calibration.c',     
            'tracking_frame_buf.c',
            'vec_utils.c',       
            'imgcoord.c',
            'image_processing.c',
            'correspondences.c',
            'segmentation.c',
            'track.c',
            'orientation.c',
            'trafo.c',
            'lsqadj.c',
            'multimed.c',
            'ray_tracing.c'
        ]
    ]
    return base_sources

# First compile vec_utils as it's a dependency
ext_mods = [
    Extension(
        "optv.vec_utils",
        sources=["optv/vec_utils.c"] + get_liboptv_sources(),
        include_dirs=[
            np.get_include(),
            'optv',
            './liboptv/include',
            './liboptv/include/optv',
            './liboptv/src'
        ],
        extra_compile_args=['-O3', '-fPIC'],
    ),
    # Then compile tracking_framebuf which depends on vec_utils
    Extension(
        "optv.tracking_framebuf",
        sources=["optv/tracking_framebuf.c"] + get_liboptv_sources(),
        include_dirs=[
            np.get_include(),
            'optv',
            './liboptv/include',
            './liboptv/include/optv',
            './liboptv/src'
        ],
        extra_compile_args=['-O3', '-fPIC'],
    )
]

setup(
    ext_modules=ext_mods,
    cmdclass={'build_ext': build_ext},
)
