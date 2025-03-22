# cython: language_level=3
# distutils: language = c

from optv.calibration cimport calibration
from optv.parameters cimport mm_np, volume_par
from optv.vec_utils cimport vec3d

cdef extern from "optv/epi.h":
    void  epi_mm_2D (double xl, double yl, calibration *cal, mm_np *mmp, 
        volume_par *vpar, vec3d out);