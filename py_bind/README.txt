
OPTV - a Python package wrapping liboptv
========================================
In this package you will find Python wrappers for accessing the OpenPTV core
library (a.k.a liboptv). The package is imported from Python as the example below shows.

Currently the wrappers include two classes: Target, which wraps around the target 
structure in liboptv; and TargetArray, which wraps around a C array of targets and 
provides Pythonic access to it. The function read_targets() is supplied, which
relies on the liboptv function of the same name and returns a TargetArray object.

The plan is to add more wrappers as other contributors of liboptv find them 
necessary and choose to add them here.


Installation
------------
This package assumes that liboptv is already installed. If it is not, see the 
instructions for installing it in the liboptv source code.

To build the wrapper, Cython must also be installed. Binary installers are 
available at www.cython.org. Linux users may simply install from the package 
manager, Windows users can get it through Python(x,y). If you have installed
openptv-python from source, then you already have Cython working.

The test suite consists of Python code that may be run automatically using the 
Nose test harness: https://nose.readthedocs.org/en/latest/#

With the dependencies installed, the optv package is installed by typing the
following command in a terminal:

  python setup.py install

Note that on many systems you will first need to obtain administrator 
privileges. On Linux the 'sudo' command is recommended.


Usage example
-------------
This code appears in a movie viewer using the optv wrappers:

  from optv.tracking_framebuf import read_targets
  ...
  def image_targets(self, frame, cam):
      tmpl = re.sub('%1', '%d', self._tmpl)
      tmpl = re.sub('%2', '', tmpl)
      targets = read_targets(tmpl % (cam + 1), frame)
  ...


