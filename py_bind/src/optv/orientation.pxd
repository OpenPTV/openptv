# cython: language_level=3
# distutils: language = c

from optv.calibration cimport calibration
from optv.parameters cimport control_par, mm_np
from optv.tracking_framebuf cimport target
from optv.vec_utils cimport vec3d
from optv.epipolar cimport epi_mm_2D

cdef extern from "optv/sortgrid.h":
    target *sortgrid(calibration *cal, control_par *cpar, int nfix, vec3d fix[], int num, int eps, target pix[])

cdef extern from "optv/orientation.h":
    ctypedef double vec2d[2]
    ctypedef struct orient_par:
        unsigned int useflag
        unsigned int ccflag, xhflag, yhflag
        unsigned int k1flag, k2flag, k3flag
        unsigned int p1flag, p2flag
        unsigned int scxflag, sheflag, interfflag

    enum:
        NPAR
    double COORD_UNUSED

    double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
        calibration* cals[], vec3d res);
    double single_cam_point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
        calibration* cals[], vec3d res);
    int raw_orient(calibration* cal, control_par *cpar, int nfix, vec3d fix[], 
        target pix[]);
    double* orient (calibration* cal_in, control_par *cpar, int nfix, 
        vec3d fix[], target pix[], orient_par *flags, double sigmabeta[20])
    orient_par* read_orient_par(char *filename)
    double weighted_dumbbell_precision(vec2d** targets, int num_targs, 
        int num_cams, mm_np *multimed_pars, calibration* cals[], 
        int db_length, double db_weight)

cdef calibration** cal_list2arr(list cals)
