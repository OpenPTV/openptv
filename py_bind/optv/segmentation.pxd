# cython: language_level=3
# distutils: language = c

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:12:36 2016

@author: yosef
"""

from optv.tracking_framebuf cimport target
from optv.parameters cimport target_par, control_par

cdef extern from "optv/segmentation.h":
    int targ_rec (unsigned char *img, target_par *targ_par, int xmin, 
        int xmax, int ymin, int ymax, control_par *cpar, int num_cam, 
        target pix[])
