
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
Run pip install openptv, which should install everything.

Building the Package
--------------------
The package has to be built from the full repository. Make sure all the dependencies
in requirements.txt are installed, then run:

python setup.py prepare   # This copies the liboptv sources and converts pyx files to C
python setup.py build     # This builds the package

You can then create source distributions and binary wheels:

python setup.py sdist bdist_wheel

You can upload them to your favorite repository, as they reside in the dist subdirectory.

Note: You need to build wheels for each platform you want to support.

On Windows, you must install the Visual C++ Compiler for Python 2.7. It can be found here:
https://www.microsoft.com/en-us/download/details.aspx?id=44266

You will need to build the package from a Visual C++ for Python Command Prompt.


Testing the installation
------------------------
Regardless of the system you installed on, the shell commands are the same:

  cd tests/
  nosetests *

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


