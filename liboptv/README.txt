
liboptv - a library of openptv support code, for use in user-facing code.
-------------------------------------------------------------------------

This is a library - you need to build it and link to it in your own project.

Below are instructions for building it and making it ready for use, either from
a source distribution (a tarball with all necessary files) or directly from a 
Git repository.

When the package is installed correctly, you can reference it in your code by 
including files from the optv directory under the standard include path. For
example,

  #include <optv/tracking_frame_buf.h>

To build your program you also link it with liboptv. On gcc, one adds the flag
-loptv to the command line. Other compilers and IDEs have their own 
instructions for adding libraries; consult your IDE/compiler manual for the 
details.


Installing on Linux/Mac 
======================= 
Installing directly from the source tree in Git is fast and simple, using CMake. 
Before installation first we need to make sure that the dependencies are 
installed.

1. The tests of liboptv depend on the Check framework. You can either get it 
   from your package manager (on Ubuntu, 'sudo apt-get install check'), or
   get the source package  from check.sourceforge.net and install according 
   to the instructions in the Check package. Typically you would need to run 
   the following commands in a shell:

     $ ./configure
     $ make
     $ make install

2. The build instructions for liboptv are processed with CMake. Again, this 
   is available either from your package manager on Linux, or from cmake.org.
   after installing CMake, installing liboptv is a simple matter of running
   the following commands in the liboptv/ directory:
   
     $ cmake
     $ make
     $ make verify # optional, runs the test suite.
     $ make install

The install process will put the header files and libraries in default system
paths (on linux, /usr/local/include and /usr/local/lib respectively), so that 
you can use the optv library in your code without further modifications. You 
can run cmake with parameters that would change the install locations (see
below on the Windows install process). If you do so, remmember to make sure 
that your compiler knows the path to the installed location.


Installing on Windows
=====================
Unlike Linux and MacOS, which both implement POSIX standards (the Unix 
standards base) and contain a default C build environment, Windows has 
no such environment. Choices range from the bare-bones Windows SDK, whose 
compiler is based on an outdated verion of C, to Visual Studio, a commercial 
product with its own IDE and build toolchain. Aside from being costly and 
proprietary, building with these compilers introduces compatibility problems
with other programs. For example, building Python modules with VC 2010 from
the Windows SDK fails because Python was build with VC 2008.

All this setup is here to justify the fact that build instructions here are
for the MingW compiler and the MSYS package of Unix tools for Windows. After a
hard process of trial and error this was found to be the easiest, most 
compatible solution.

The MSYS package provides the GCC compiler (MinGW), a Bash command-line shell 
and Unix build tools for Windows. It can be found here:
http://www.mingw.org/wiki/MSYS

Use the mingw-get-setup method of instalation. During the installation you will
be asked to choose subpackages. If you don't know what you're doing, choose
everything. 

After installing MSYS MinGW according to the instructions on the MSYS site, you 
will have a MinGW shell or MSYS shell in your start menu. Future instructions 
assume that this shell is used. the installation instructions on the MSYS page 
given above list some more steps you can and should do, so follow that page 
carefully. In particular, don't forget to create the fstab file as instructed 
there.

Installing Check
~~~~~~~~~~~~~~~~
The tests of liboptv depend on the Check framework. You can get it from 
check.sourceforge.net

Some versions have problems with Windows. Version 0.9.8 is known to work. You can 
get version 0.9.14 to work by editing lib/libcompat.h and commenting out or 
removing lines 147-151.

Installing Check is done roughly in the same way as on Linux, in the MSYS 
shell:

     $ ./configure --prefix /usr
     $ make
     $ make install

However, it is important to note where the install actually lands so that we can 
help CMake find it. The Check library would be installed under the MSYS tree 
which was set up when installing MSYS. The above installation is in what MSYS 
refers to as /usr. If your MSYS is installed on C:\MinGW, then Check would then 
be in ``C:\MinGW\msys\1.0\lib\`` and ``C:\MinGW\msys\1.0\include`` or a similar 
path. Make sure to verify this.

Installing liboptv
~~~~~~~~~~~~~~~~~~
Now that Check is installed, installing liboptv is relatively straightforward.
Since you are reading this file, you already have that package. enter the 
liboptv/ subdirectory, create a directory under it called build/, and change
into it.

For processing of build instructions, install CMake, from cmake.org.

Now, in the Build directory, initialize cmake with the following command:
    
    $ cmake ../ -G "Unix Makefiles" -i

CMake will then ask you some questions about your system (accepting the 
defaults is usually ok). Now and at any future step, you can erase the 
contents of the build/ directory and start over. You can also regenerate
makefiles with a simple 'cmake ../' in a working build directory, since 
CMake caches values you set before.

Now that CMake is initialized, a command to generate Makefiles with all
paths told in advance, would be

    $  cmake ../ -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_PREFIX_PATH=/c/MinGW/msys/1.0/

Note the path where Check was installed is specified, and be sure to adjust it
if it is a different path in your system.

Now to build and install liboptv, type

    $ make
    $ make install

This would install liboptv in what MSYS refers to as /usr, which is 
C:\MinGW\msys\1.0\ on my system. Any further program that is built using MSYS
looks for this path by default so no further adjustment is necessary for using
liboptv in your program, other than adding the include and link directives
specified above. 

However, on run time it appears that the pyd file we just installed looks for
the accompanying DLL that was installed alongside it. Windows wants this DLL
to be in the PATH it searches for executables. So the last step of installing
on Windows is to modify the PATH environment variable so that it lists the 
place where the liboptv DLL is installed (in our example, this would be 
C:\MinGW\msys\1.0\lib). This can be done by right-clicking Computer on the 
start menu, choosing Properties -> Advanced system settings -> click Environment 
Variables -> edit the PATH variable on the bottom list and add the DLL's location,
separated by a semicolon (;) charachter from the directories already listed.
 

