OpenPTV - framework for particle tracking velocimetry
========================================

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


Installation
-----------

On Mac OS X and Linux (without `CC="gcc -arch 386"`):

1. Download the **Check** unit testing framework from <http://check.sourceforge.net>. Note: use version 0.9.8 (latest 0.9.9 apparently has some bug). Run:

		./configure CC="gcc -arch i386"  
		make  
		make check  
		make install  

2. Download the `liboptv` tarball or clone it using Git <http://github.com/OpenPTV/openptv>. Install using `autoreconf`, `automake` tools:


		 cd liboptv  
		 mkdir m4  
		 autoreconf --install  
		 ./configure CC="gcc -arch i386"  
		 make  
		 make check  
		 make install  


	- If you experience some error messages, try the following:


		    autoreconf
		    automake --add-missing
		    libtoolize 
		    autoreconf
		    automake --add-missing
		    autoreconf --install
		  

Ask for help on the community mailing-list: `openptv@googlegroups.com`


