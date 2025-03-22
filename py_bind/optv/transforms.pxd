# cython: language_level=3
# distutils: language = c

from optv.parameters cimport control_par
from optv.calibration cimport ap_52, calibration

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
    
    void correct_brown_affine_exact(double x, double y, ap_52 ap, 
    double *x1, double *y1, double tol)
    
    void flat_to_dist(double flat_x, double flat_y, calibration *cal, 
        double *dist_x, double *dist_y)
    void dist_to_flat(double dist_x, double dist_y, calibration *cal,
        double *flat_x, double *flat_y, double tol)