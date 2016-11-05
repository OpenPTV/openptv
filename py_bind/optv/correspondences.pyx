# -*- coding: utf-8 -*-
"""
Implementation of bindings for correspondences and related data structures.

Created on Fri Oct 28 13:46:39 2016

@author: yosef
"""

from libc.stdlib cimport malloc, free

from optv.transforms cimport pixel_to_metric, dist_to_flat
from optv.parameters cimport ControlParams
from optv.calibration cimport Calibration
from optv.tracking_framebuf cimport target

cdef class MatchedCoords:
    """
    Keeps a block of 2D flat coordinates, each with a "point number", the same
    as the number on one ``target`` from the block to which this block is kept
    matched. This block is x-sorted.
    
    NB: the data is not meant to be directly manipulated at this point. The 
    coord_2d arrays are most useful as intermediate objects created and 
    manipulated only by other liboptv functions. Although one can imagine a 
    use case for direct manipulation in Python, it is rare and supporting it 
    is a low priority.
    """
    
    def __init__(
        self, TargetArray targs, ControlParams cpar, 
        Calibration cal, double tol=0.00001, reset_numbers=True):
        """
        Allocates and initializes the memory, including coordinate conversion 
        and sorting.
        
        Arguments:
        TargetArray targs - the TargetArray to be converted and matched.
        ControlParams cpar - parameters of image size etc. for conversion.
        Calibration cal - representation of the camera parameters to use in
            the flat/distorted transforms.
        double tol - optional tolerance for the lens distortion correction 
            phase, see ``optv.transforms``.
        reset_numbers - if True (default) numbers the targets too, in their 
            current order. This shouldn't be necessary since all TargetArray
            creators number the targets, but this gets around cases where they
            don't.
        """
        cdef:
            target *targ
            int  num_targs = len(targs)
        
        self.buf = <coord_2d *> malloc(num_targs * sizeof(coord_2d))
        if self.buf == NULL:
            raise MemoryError("could not allocate matched-coordinates array.")
        
        for tnum in range(num_targs):
            targ = &(targs._tarr[tnum])
            if reset_numbers:
                targ.pnr = tnum
            
            pixel_to_metric(
                &(self.buf[tnum].x), &(self.buf[tnum].y), targ.x, targ.y, 
                cpar._control_par)
            dist_to_flat(
                self.buf[tnum].x, self.buf[tnum].y, cal._calibration,
                &(self.buf[tnum].x), &(self.buf[tnum].y), tol)
            self.buf[tnum].pnr = targ.pnr
        
        quicksort_coord2d_x(self.buf, num_targs)
    
    
    def __dealloc__(self):
        free(self.buf)
