"""
not so much a binding for the epipolar code, as a more general implementation
of epipolar curve finding using a similar algorithm to that in epi.c.

Created on Sun Mar 19 19:42:41 2017

@author: yosef
"""
import numpy as np
cimport numpy as np

from optv.calibration cimport Calibration, calibration
from optv.parameters cimport ControlParams, VolumeParams, mm_np
from optv.vec_utils cimport vec3d
from optv.transforms cimport metric_to_pixel, pixel_to_metric, dist_to_flat
from optv.imgcoord cimport img_coord

cdef extern from "optv/ray_tracing.h":
    void ray_tracing(double x, double y, calibration* cal, mm_np mm,
        double X[3], double a[3]);

cdef extern from "optv/multimed.h":
    void move_along_ray(double glob_Z, vec3d vertex, vec3d direct, vec3d out)

def epipolar_curve(np.ndarray[ndim=1, dtype=np.float64_t] image_point,
    Calibration origin_cam, Calibration project_cam, int num_points,
    ControlParams cparam, VolumeParams vparam):
    """
    Get the points lying on the epipolar line from one camera to the other, on
    the edges of the observed volume. Gives pixel coordinates.
    
    Assumes the same volume applies to all cameras.
    
    Arguments:
    np.ndarray[ndim=1, dtype=pos_t] image_point - the 2D point on the image
        plane of the camera seeing the point. Distorted pixel coordinates.
    Calibration origin_cam - current position and other parameters of the 
        camera seeing the point.
    Calibration project_cam - current position and other parameters of the 
        cameraon which the line is projected.
    int num_points - the number of points to generate along the line. Minimum
        is 2 for both endpoints.
    ControlParams cparam - an object holding general control parameters.
    VolumeParams vparam - an object holding observed volume size parameters.
    
    Returns:
    line_points - (num_points,2) array with projection camera image coordinates
        of points lying on the ray stretching from the minimal Z coordinate of 
        the observed volume to the maximal Z thereof, and connecting the camera 
        with the image point on the origin camera.
    """
    cdef:
        np.ndarray[ndim=2, dtype=np.float64_t] line_points
        vec3d vertex, direct, pos
        int pt_ix
        double Z
        double *x
        double *y
        double img_pt[2]
    
    line_points = np.empty((num_points, 2))
    
    # Move from distorted pixel coordinates to straight metric coordinates.
    pixel_to_metric(img_pt, img_pt + 1, image_point[0], image_point[1], 
        cparam._control_par)
    dist_to_flat(img_pt[0], img_pt[1], 
        origin_cam._calibration, img_pt, img_pt + 1, 0.00001)
    
    ray_tracing(img_pt[0], img_pt[1], origin_cam._calibration,
        cparam._control_par.mm[0], vertex, direct)
    
    for pt_ix, Z in enumerate(np.linspace(vparam._volume_par.Zmin_lay[0], 
        vparam._volume_par.Zmax_lay[0], num_points)):
        
        x = <double *>np.PyArray_GETPTR2(line_points, pt_ix, 0)
        y = <double *>np.PyArray_GETPTR2(line_points, pt_ix, 1)
        
        move_along_ray(Z, vertex, direct, pos)
        img_coord(pos, project_cam._calibration, cparam._control_par.mm, x, y)
        metric_to_pixel(x, y, x[0], y[0], cparam._control_par)
    
    return line_points