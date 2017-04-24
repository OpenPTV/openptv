from optv.calibration cimport calibration
from optv.parameters cimport control_par, mm_np
from optv.tracking_framebuf cimport target
from optv.vec_utils cimport vec3d

cdef extern from "optv/sortgrid.h":
    target *sortgrid(calibration *cal, control_par *cpar, int nfix, vec3d fix[], int num, int eps, target pix[])

cdef extern from "optv/orientation.h":
    ctypedef double vec2d[2]

    double COORD_UNUSED

    double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
        calibration* cals[], vec3d res);

cdef calibration** cal_list2arr(list cals)