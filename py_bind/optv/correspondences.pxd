# cython: language_level=3
# distutils: language = c

# -*- coding: utf-8 -*-
"""
Cython definitions for correspondences and related data structures.

Created on Fri Oct 28 13:47:26 2016

@author: yosef
"""

from optv.tracking_framebuf cimport TargetArray, frame
from optv.parameters cimport volume_par, control_par
from optv.calibration cimport calibration

# For the life of me, I don't know why find_candidate and its related coord_2d 
# should be in epi.h, but it's there and I'm not moving it right now.
cdef extern from "optv/epi.h":
    ctypedef struct coord_2d:
        int pnr
        double x, y

cdef extern from "optv/correspondences.h":
    ctypedef struct n_tupel:
        int p[4]
    
    void quicksort_coord2d_x(coord_2d *crd, int num)
    n_tupel* corresp "correspondences" (frame *frm, coord_2d **corrected, 
        volume_par *vpar, control_par *cpar, calibration **calib,
        int match_counts[])
    
cdef class MatchedCoords:
    cdef coord_2d *buf
    cdef int _num_pts
