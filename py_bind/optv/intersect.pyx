from __future__ import division

# from libc.stdlib cimport malloc , free

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "../liboptv/include/intersect.h":
    void intersect_rt (double pos1[3], double vec1[3], double pos2[3], double vec2[3], 
    double X[3])


# @cython.boundscheck(False)
# @cython.wraparound(False)
def py_intersect(np.ndarray[np.float_t, ndim=1] pos1, np.ndarray[np.float_t, ndim=1] vec1,\
np.ndarray[np.float_t, ndim=1] pos2, np.ndarray[np.float_t, ndim=1] vec2):
    cdef np.ndarray[np.float_t, ndim=1] X = np.empty(3,dtype=np.float)
    intersect_rt(&pos1[0], &vec1[0], &pos2[0], &vec2[0], &X[0])
    print X
    return X