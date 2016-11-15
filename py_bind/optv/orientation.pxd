from optv.calibration cimport calibration
from optv.parameters cimport control_par
from optv.tracking_framebuf cimport target
from optv.vec_utils cimport vec3d

cdef extern from "optv/sortgrid.h":
    target *sortgrid(calibration *cal, control_par *cpar, int nfix, vec3d fix[], int num, int eps, target pix[])
