
A simple 4-camera movie viewer using liboptv
============================================
The viewer app in this directory is a small example of using liboptv through
its Python bindings. It shows a series of images whose name indicates the frame
number and camera number, with play/pause, advance and rewind buttons. If the
directory containing the movie data has target files in the format understood
by liboptv, with the same name as the images except for a '_targets' suffix,
then the viewer can mark the targets with red dots on user selection.

Usage
=====
Installing the viewer is not required, it is run from this directory directly.
It is only required that the dependencies are installed - i.e. liboptv and
its Python bindings, which are found, respectively, in the liboptv/ and
py_bind/ subdirectories of the openptv source package.

Running the viewer with the -h switch gives instructions on how to use it:

  $ python movie_browser.py -h
  usage: movie_browser.py [-h] [--rate RATE] template first last
  
  positional arguments:
    template     Temlate for image file name, containing %1 where camera number
                 should be inserted, and %2 where frame number should be
                 inserted. Example: my_scene/cam%1.%2
    first        Number of first frame
    last         Number of last frame
  
  optional arguments:
    -h, --help   show this help message and exit
    --rate RATE  Frame rate in frames per second

As an example, this runs a series of files with a frame rate of 60 fps:

  $ python movie_browser.py ~/scene/cam%1_Scene18_%2 2380 2450 --rate 60

In the example, a directory called scene/ in my home directory has TIFF files
with names like cam3_Scene18_2400, and target files with names like
cam3_Scene18_2400_targets. The '--rate' switch sets the fps, and is optional.

The buttons in the control row are as follows:
* rewind
* back one frame
* play/pause (first click to play, second click to pause)
* forward one frame
* forward to end of movie.

In the row below, two option boxes are initially greyed out. Click to enable
(box turns green). The first, 'continuous replay', rewinds the movie when the
last frame is reached, and plays it again. The second option, 'show targets',
paints all recognized targets, as read from the frame's _targets files, in red.

