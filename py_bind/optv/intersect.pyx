from __future__ import division

# from libc.stdlib cimport malloc , free

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "../liboptv/include/intersect.h":
    void intersect_rt (double pos1[3], double vec1[3], double pos2[3], double vec2[3], \
    double *X, double *Y, double *Z)


# @cython.boundscheck(False)
# @cython.wraparound(False)
def py_intersect(np.ndarray[np.double_t, ndim=1] pos1, np.ndarray[np.double_t, ndim=1] vec1,\
np.ndarray[np.double_t, ndim=1] pos2, np.ndarray[np.double_t, ndim=1] vec2, \
np.ndarray[np.double_t, ndim=1] X, np.ndarray[np.double_t, ndim=1] Y, \
np.ndarray[np.double_t, ndim=1] Z):
    intersect_rt(<double*>pos1.data, <double*>vec1.data, <double*>pos2.data, <double*>vec2.data, \
    <double*>X.data,<double*>Y.data,<double*>Z.data)
    return X,Y,Z