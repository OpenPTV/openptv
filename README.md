OpenPTV - framework for particle tracking velocimetry
=====================================================

This is the code collection of the OpenPTV project - an effort to create
common code for different aspects of the Particle Tracking Velocimetry
method. The code is required to meet the community standards for quality, and
all code here is therefore peer reviewed.

The quality standards are decided upon open discussion on the community 
mailing-list, 

    openptv@googlegroups.com

The peer-review process happens in the open on the same mailing list.


How to help
-----------
To contribute code: fork this repository, build on your fork an orderly branch
with your changes, then create a pull request on Github and inform the mailing
list, or post a patch series to the mailing list. Be prepared to answer 
questions and amend your code to satisfy reviewer comments. Be mindful of the
agreed coding standards (whose URL will be soon updated here).

To follow and participate in the technical discussion: join the mailing list
through Google Groups.


Instalation
-----------

On Mac OS X:

1. Download check unit testing framework from <http://check.sourceforge.net>. Note: use ver. 0.9.8 (latest 0.9.9 has some bug)

    ./configure CC="gcc -arch i386"  
    make  
    make check  
    make install  

2. Download the liboptv or clone it using Git. Install using autoconf, automake tools:


      cd liboptv  
      mkdir m4  
      autoreconf --install  
      ./configure CC="gcc -arch i386"  
      make  
      make check  
      make install  


If you experience some error messages, try the following:


    autoreconf
    automake --add-missing
    libtoolize 
    autoreconf
    automake --add-missing
    autoreconf --install


