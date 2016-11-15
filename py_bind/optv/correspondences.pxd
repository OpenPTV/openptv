# -*- coding: utf-8 -*-
"""
Cython definitions for correspondences and related data structures.

Created on Fri Oct 28 13:47:26 2016

@author: yosef
"""

from optv.tracking_framebuf cimport TargetArray

# For the life of me, I don't know why find_candidate and its related coord_2d 
# should be in epi.h, but it's there and I'm not moving it right now.
cdef extern from "optv/epi.h":
    ctypedef struct coord_2d:
        int pnr
        double x, y

cdef extern from "optv/correspondences.h":
    void quicksort_coord2d_x(coord_2d *crd, int num)

cdef class MatchedCoords:
    cdef coord_2d *buf
    cdef int _num_pts
