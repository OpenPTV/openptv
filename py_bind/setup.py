# -*- coding: utf-8 -*-

import os
import shutil
import sys
import glob
import numpy
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

class PrepareCommand(Command):
    """Prepare the C sources by copying them from liboptv and converting pyx to C"""

    description = "Prepare C sources and Cython files"
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
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
            shutil.copy(h_file, 'liboptv/include/')

            # # Also copy headers to the root liboptv directory for compatibility
            # dest = os.path.join('liboptv', os.path.basename(h_file))
            # shutil.copy(h_file, dest)

        # Convert pyx to C

        cythonize(['optv/*.pyx'], compiler_directives={'language_level': '3'})


class BuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.include_dirs.append(numpy.get_include())

    def run(self):
        if not os.path.exists('./liboptv') or not glob.glob('./optv/*.c'):
            print('You must run setup.py prepare before building the extension', file=sys.stderr)
            raise Exception('You must run setup.py prepare before building the extension')
        super().run()


def get_liboptv_sources():
    return glob.glob('./liboptv/src/*.c')


def mk_ext(name, files):
    extra_compile_args = []
    extra_link_args = []

    # Create optv directory structure for include files
    os.makedirs('optv/optv', exist_ok=True)
    for header in glob.glob('liboptv/include/*.h'):
        header_name = os.path.basename(header)
        target_path = os.path.join('optv/optv', header_name)
        if not os.path.exists(target_path):
            # Always copy the file for consistency across platforms
            shutil.copy(header, target_path)

    if not sys.platform.startswith('win'):
        extra_compile_args.extend(['-Wno-cpp', '-Wno-unused-function'])
        extra_link_args.extend(['-Wl,-rpath,$ORIGIN'])
    else:
        extra_compile_args.append('/W4')
        extra_compile_args.append('/std:c11')
        extra_compile_args.append('/D_CRT_SECURE_NO_WARNINGS')

    include_dirs = [
        numpy.get_include(),
        './liboptv/include/',
        './optv/',
    ]

    # Add absolute paths for Windows
    if sys.platform.startswith('win'):
        include_dirs.extend([
            os.path.abspath('./liboptv/include/'),
            os.path.abspath('./optv/'),
        ])

    return Extension(
        name,
        files + get_liboptv_sources(),
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


def get_extensions():
    extensions = []
    for pyx_file in glob.glob('optv/*.pyx'):
        module = os.path.splitext(pyx_file)[0].replace(os.path.sep, '.')
        c_file = os.path.splitext(pyx_file)[0] + '.c'
        extensions.append(mk_ext(module, [c_file]))
    return extensions


if __name__ == "__main__":
    setup(
        cmdclass={
            'build_ext': BuildExt,
            'prepare': PrepareCommand,
        },
        ext_modules=get_extensions(),
        include_package_data=True,
        package_data={
            'optv': ['*.pxd', '*.c', '*.h'],
        },
    )
