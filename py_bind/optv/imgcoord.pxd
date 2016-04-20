from optv.calibration cimport calibration
from optv.parameters cimport mm_np
from optv.vec_utils cimport vec3d

cdef extern from "optv/imgcoord.h":
    void c_image_coord "img_coord"(vec3d pos,
                                 calibration * cal,
                                 mm_np * mm,
                                 double * x,
                                 double * y) 
    void c_flat_image_coord "flat_image_coord"(vec3d pos,
                                               calibration * cal,
                                               mm_np * mm,
                                               double * x,
                                               double * y)