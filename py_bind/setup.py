# -*- coding: utf-8 -*-

from distutils.core import setup
import setuptools
import os
import shutil
import sys
import glob
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import numpy as np

class PrepareCommand(setuptools.Command):
    description = "Copy the liboptv sources and convert pyx files to C before building"
    user_options = [('liboptv-dir=', None, 'Path for liboptv, default is "../liboptv"')]

    def initialize_options(self):
        self.liboptv_dir = False

    def finalize_options(self):
        if not self.liboptv_dir:
            self.liboptv_dir = '../liboptv'

    def run(self):
        self.copy_source_files()
        self.convert_to_c()

    def copy_source_files(self):
        if not os.path.exists(self.liboptv_dir):
            print('liboptv does not exist at %s. You must run setup.py prepare from with the full liboptv repository' % self.liboptv_dir,
                  file=sys.stderr)
            raise Exception('liboptv does not exist at %s' % self.liboptv_dir)
        
        print('Copying the liboptv source files from %s...' % self.liboptv_dir)
        if os.path.exists('./liboptv'):
            shutil.rmtree('./liboptv')
        os.makedirs('./liboptv')
        
        # Create all necessary directories
        os.makedirs('./liboptv/include/optv', exist_ok=True)
        os.makedirs('./liboptv/src', exist_ok=True)
        
        # Copy all header files from include and its subdirectories
        for root, dirs, files in os.walk(os.path.join(self.liboptv_dir, 'include')):
            for file in files:
                if file.endswith(('.h', '.hpp')):
                    src_path = os.path.join(root, file)
                    # Get relative path from include dir
                    rel_path = os.path.relpath(src_path, os.path.join(self.liboptv_dir, 'include'))
                    dst_path = os.path.join('liboptv/include/optv', rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    print(f'Copied header: {rel_path}')
        
        # Copy all source files
        for root, dirs, files in os.walk(os.path.join(self.liboptv_dir, 'src')):
            for file in files:
                if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, os.path.join(self.liboptv_dir, 'src'))
                    dst_path = os.path.join('liboptv/src', rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    print(f'Copied source: {rel_path}')

    def convert_to_c(self):
        print('Converting pyx files to C sources...')
        try:
            import subprocess
            subprocess.check_call([sys.executable, 'cythonize.py'])
        except subprocess.CalledProcessError as e:
            print(f'Error converting pyx files to C: {str(e)}', file=sys.stderr)
            raise


class BuildExt(build_ext):
    def run(self):
        if not os.path.exists('./liboptv') or not glob.glob('./optv/*.c'):
            print('You must run setup.py prepare before building the extension', file=sys.stderr)
            raise Exception('You must run setup.py prepare before building the extension')
        self.add_include_dirs()
        super(BuildExt, self).run()

    @staticmethod
    def get_numpy_include_dir():
        import numpy
        return numpy.get_include()

    def add_include_dirs(self):
        np_include_dir = BuildExt.get_numpy_include_dir()
        include_dirs = [np_include_dir, '.', './liboptv/include', './liboptv/include/optv']
        for extension in self.extensions:
            extension.include_dirs = include_dirs


def get_liboptv_sources():
    """Get all required C source files"""
    base_sources = [
        './liboptv/src/parameters.c',
        './liboptv/src/calibration.c',
        './liboptv/src/tracking_frame_buf.c',
        './liboptv/src/ray_tracing.c',
        './liboptv/src/imgcoord.c',
        './liboptv/src/image_processing.c',
        './liboptv/src/correspondences.c',
        './liboptv/src/segmentation.c',
        './liboptv/src/track.c',
        './liboptv/src/orientation.c',
        './liboptv/src/trafo.c',
        './liboptv/src/lsqadj.c',
        './liboptv/src/vec_utils.c',
        './liboptv/src/multimed.c',
    ]
    
    # Verify all files exist
    for source in base_sources:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Required source file not found: {source}")
    
    return base_sources


def mk_ext(name, sources):
    return Extension(
        name,
        sources=sources + get_liboptv_sources(),
        include_dirs=[
            np.get_include(),
            'optv',
            './liboptv/include',
            './liboptv/include/optv',
            './liboptv/src'
        ],
        extra_compile_args=[
            '-O3',
            '-fPIC',
            '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
            '-Wall'  # Add warnings to help debug
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_link_args=['-Wl,-rpath,$ORIGIN']
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
    cmdclass={
        'build_ext': BuildExt,
        'prepare': PrepareCommand,
    },
    ext_modules=ext_mods,
)
