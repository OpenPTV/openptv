# To Do list

If you're still not sure what to do after reading this, and just as a generally
good idea, you should contact the mailing list and tell us that you're looking to
help. The list address is <openptv@googlegroups.com>

## liboptv or all the following
1. change the parameters files to INF or similar ASCII format. e.g. http://docs.python.org/2/library/configparser.html

### Small tasks ###
1. Identify some small, discrete functionality of 3D-PTV which can be enclosed in a 
function, and write just this function. You may depend on code that's 
already in liboptv, but not more. Examples:
    1.  Image processing, particle detection.
    2.  Generating an epipolar line.
    3.  Deciding which targets are near enough to a given epipolar line.

2. Call a function from liboptv in your C/Matlab/Python code, showing to us and
yourself that it's working. Contribute the bindings you used if you did.

### Larger tasks ###
1. Move one of the larger blocks of functionality from C-Tcl/Tk. This should be
done gradually, ask on the mailing list and we'll come up with a good plan.
2. Write bindings for some language.
3. Show that liboptv can be done in C++ and used in C.


## Python
1. Try Cmake on top of autoconf/automake to have a unified build system

## Matlab
1. Tests

## Post-processing
1. Decide about the platform/language
2. 

## Misc
