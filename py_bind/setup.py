# -*- coding: utf-8 -*-
from __future__ import print_function
from distutils.core import setup
import setuptools
import os
import shutil
import sys
import glob
from setuptools import Extension
from setuptools.command.build_ext import build_ext


class PrepareCommand(setuptools.Command):
    # We must make some preparations before we can build the extension.
    # First, we should copy the liboptv sources to a subdirectory, so they can be included with the sdist package.
    # Second, we convert the pyx files to c files, so the package can be installed from source without requiring Cython
    description = "Copy the liboptv sources and convert pyx files to C before building"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.copy_source_files()
        self.convert_to_c()

    def copy_source_files(self):
        if not os.path.exists('../liboptv'):
            print('../liboptv does not exist. You must run setup.py prepare from with the full liboptv repository',
                  file=sys.stderr)
            raise Exception('.//liboptv does not exist')
        
        print('Copying the liboptv source files...')
        if os.path.exists('./liboptv'):
            shutil.rmtree('./liboptv')
        os.makedirs('./liboptv')
        shutil.copytree('../liboptv/include', 'liboptv/include/optv')
        shutil.copytree('../liboptv/src', 'liboptv/src')

    def convert_to_c(self):
        print('Converting pyx files to C sources...')
        pyx_files = glob.glob('./optv/*.pyx')
        for pyx in pyx_files:
            self.cython(pyx)

    def cython(self, pyx):
        from Cython.Compiler.CmdLine import parse_command_line
        from Cython.Compiler.Main import compile
        options, sources = parse_command_line(['-2', pyx])
        result = compile(sources, options)
        if result.num_errors > 0:
            print('Errors converting %s to C' % pyx, file=sys.stderr)
            raise Exception('Errors converting %s to C' % pyx)
        self.announce('Converted %s to C' % pyx)


class BuildExt(build_ext, object):
    def run(self):
        if not os.path.exists('./liboptv') or not glob.glob('./optv/*.c'):
            print('You must run setup.py prepare before building the extension', file=sys.stderr)
            raise Exception('You must run setup.py prepare before building the extension')
        self.add_include_dirs()
        super(BuildExt, self).run()

        # We inherite from object to make super() work, see here: https://stackoverflow.com/a/18392639/871910

    @staticmethod
    def get_numpy_include_dir():
        # Get the numpy include directory, adapted from the following  RLs:
        # https://www.programcreek.com/python/example/60953/__builtin__.__NUMPY_SETUP__
        # https://github.com/astropy/astropy-helpers/blob/master/astropy_helpers/utils.py
        if sys.version_info[0] >= 3:
            import builtins
            if hasattr(builtins, '__NUMPY_SETUP__'):
                del builtins.__NUMPY_SETUP__
            import imp
            import numpy
            imp.reload(numpy)
        else:
            import __builtin__
            if hasattr(__builtin__, '__NUMPY_SETUP__'):
                del __builtin__.__NUMPY_SETUP__
            import numpy
            reload(numpy)

        try:
            return numpy.get_include()
        except AttributeError:
            return numpy.get_include_dir()

    def add_include_dirs(self):
        # All the Extension objects do not have their include_dir specified, we add it here as it requires
        # importing numpy, which we do not want to do unless build_ext is really running.
        # This allows pip to install numpy as it processes dependencies before building extensions
        np_include_dir = BuildExt.get_numpy_include_dir()
        include_dirs = [np_include_dir, '.', './liboptv/include', './liboptv/include/optv']

        for extension in self.extensions:  # We dug into setuptools and distutils to find the properties to change
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
    # Do not specify include dirs, as they require numpy to be installed. Add them in BuildExt
    return Extension(name, files + get_liboptv_sources())


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
    version='0.2.5',
    install_requires=[
        'numpy>=1.16.1',
        'pyyaml',
    ],
    setup_requires=['numpy>=1.16.1'],
)
