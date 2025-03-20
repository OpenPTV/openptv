# Cython declarations for parameters.h
from libc.stdlib cimport malloc, free

cdef extern from "optv/parameters.h":
    ctypedef struct target_par:
        int discont
        int gvthres[4]
        int nnmin, nnmax
        int nxmin, nxmax
        int nymin, nymax
        int sumg_min
        int cr_sz

    ctypedef struct mm_np:
        double n1
        double n2[3]
        double n3
        double d[3]
    
    ctypedef struct control_par:
        int num_cams
        int hp_flag
        int allCam_flag
        int tiff_flag
        int imx
        int imy
        double pix_x
        double pix_y
        int chfield
        mm_np *mm
    
    # C function declarations
    control_par* new_control_par(int num_cams)
    void free_control_par(control_par* cp)
    int compare_control_par(control_par *cp1, control_par *cp2)
    target_par* read_target_par(char *filename)
    int compare_target_par(target_par *targ1, target_par *targ2)

# Cython class declarations
cdef class MultimediaParams:
    cdef mm_np* _mm_np
    cdef mm_np* get_mm_np(self) nogil
    cdef void set_mm_np(self, mm_np* other) nogil

cdef class ControlParams:
    cdef:
        control_par* _control_par
        MultimediaParams _multimedia_params

cdef class TargetParams:
    cdef target_par* _targ_par
