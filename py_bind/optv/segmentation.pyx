# -*- coding: utf-8 -*-
"""
Bindings for image segmentation / target recognition routins in liboptv.

Created on Thu Aug 18 16:22:48 2016

@author: yosef
"""
from libc.stdlib cimport calloc, realloc, free
from libc.stdio cimport printf
import numpy as np
cimport numpy as np
np.import_array()


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint8

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

from optv.parameters cimport TargetParams, ControlParams
from optv.tracking_framebuf cimport TargetArray

def target_recognition(np.ndarray[np.uint8_t, ndim=2] img, TargetParams tpar, int cam, 
    ControlParams cparam, subrange_x=None, subrange_y=None):
    """
    Detects targets (contiguous bright blobs) in an image.
    
    Limited to ~20,000 targets per image for now. This limitation comes from
    the structure of underlying C code.
    
    Arguments:
    np.ndarray img - a numpy array holding the 8-bit gray image.
    TargetParams tpar - target recognition parameters s.a. size bounds etc.
    int cam - number of camera that took the picture, needed for getting
        correct parameters for this image.
    ControlParams cparam - an object holding general control parameters.
    subrange_x - optional, tuple of min and max pixel coordinates to search
        between. Default is to search entire image width.
    subrange_y - optional, tuple of min and max pixel coordinates to search
        between. Default is to search entire image height.
    
    Returns:
    A TargetArray object holding the targets found.
    """
    cdef:
        TargetArray t = TargetArray()
        target *ret
        target *targs = <target *> calloc(1024*20, sizeof(target))
        int num_targs
        int xmin, xmax, ymin, ymax
    


    # Set the subrange (to default if not given):
    if subrange_x is None:
        xmin, xmax = 0, cparam._control_par[0].imx
    else:
        xmin, xmax = subrange_x
    
    if subrange_y is None:
        ymin, ymax = 0, cparam._control_par[0].imy
    else:
        ymin, ymax = subrange_y
    
    if img.shape[0] != ymax or img.shape[1] != xmax:
        raise ValueError("dimensions are not correct")

    assert img.dtype == DTYPE

    # The core liboptv call:
    num_targs = targ_rec(<unsigned char *>img.data, tpar._targ_par, 
        xmin, xmax, ymin, ymax, cparam._control_par, cam, targs)
    
    # Fit the memory size snugly and generate the Python return value.
    ret = <target *>realloc(targs, num_targs * sizeof(target))
    if ret == NULL:
        free(targs)
        raise MemoryError("Failed to reallocate target array.")
    
    t.set(ret, num_targs, 1)
    return t
