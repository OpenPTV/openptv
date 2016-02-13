
from optv.parameters cimport control_par
from optv.calibration cimport ap_52

cdef extern from "optv/trafo.h":
    void pixel_to_metric(double * x_metric
                         , double * y_metric
                         , double x_pixel
                         , double y_pixel
                         , control_par * parameters);
    void metric_to_pixel(double * x_pixel
                         , double * y_pixel
                         , double x_metric
                         , double y_metric
                         , control_par * parameters);
    void correct_brown_affin (double x
                         , double y
                         , ap_52 ap
                         , double * x1
                         , double * y1);                        
    void distort_brown_affin (double x
                         , double y
                         , ap_52 ap
                         , double * x1
                         , double * y1);