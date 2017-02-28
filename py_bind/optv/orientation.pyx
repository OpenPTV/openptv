import numpy as np
cimport numpy as np

ctypedef np.float64_t pos_t
from libc.stdlib cimport calloc

from optv.tracking_framebuf cimport TargetArray
from optv.calibration cimport Calibration
from optv.parameters cimport ControlParams

def match_detection_to_ref(Calibration cal,
                           np.ndarray[ndim=2, dtype=pos_t] ref_pts,
                           TargetArray img_pts,
                           ControlParams cparam,
                           eps=25):
    """
    Creates a TargetArray where the targets are those for which a point in the
    projected reference is close enough to be considered a match, ordered by 
    the order of corresponding references, with "empty targets" for detection 
    points that have no match.
    
    Each target's pnr attribute is set to the index of the target in the array, 
    which is also the index of the associated reference point in ref_pts. 
    
    Arguments:
    Calibration cal - position and other parameters of the camera.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known 
        positions of the selected 2D points found on the image.
    TargetArray img_pts - detected points to match to known 3D positions.
    ControlParams cparam - an object holding general control parameters.
    int eps - pixel radius of neighbourhood around detection to search for
        closest projection.
    
    Returns:
    TargetArray holding the sorted targets.
    """

    if len(img_pts) < len(ref_pts):
        raise ValueError('Must have at least as many targets as ref. points.')

    cdef:
        vec3d *ref_coord
        target *sorted_targs
        TargetArray t = TargetArray()

    ref_pts = np.ascontiguousarray(ref_pts)
    ref_coord = <vec3d *> ref_pts.data

    sorted_targs = sortgrid(cal._calibration, cparam._control_par,
                            len(ref_pts), ref_coord, len(img_pts), eps, img_pts._tarr)

    t.set(sorted_targs, len(ref_pts), 1)
    return t

cdef calibration** cal_list2arr(list cals):
    """
    Allocate a C array with C calibration objects based on a Python list with
    Python Calibration objects.
    """
    cdef:
        calibration ** calib
        int num_cals = len(cals)

    calib = <calibration **> calloc(num_cals, sizeof(calibration *))
    for cal in range(num_cals):
        calib[cal] = (<Calibration> cals[cal])._calibration

    return calib

def point_positions(np.ndarray[ndim=3, dtype=pos_t] targets,
                    ControlParams cparam, cals):
    """
    Calculate the 3D positions of the points given by their 2D projections.

    Arguments:
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the
        camera order of ``targets``.

    Returns:
    res - (n,3) array for n points represented by their targets.
    rcm - n-length array, the Ray Convergence Measure for eachpoint.
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] res
        np.ndarray[ndim=1, dtype=pos_t] rcm
        np.ndarray[ndim=2, dtype=pos_t] targ
        calibration ** calib = cal_list2arr(cals)
        int cam, num_cams

    # So we can address targets.data directly instead of get_ptr stuff:
    targets = np.ascontiguousarray(targets)

    num_targets = targets.shape[0]
    num_cams = targets.shape[1]
    res = np.empty((num_targets, 3))
    rcm = np.empty(num_targets)

    for pt in range(num_targets):
        targ = targets[pt]
        rcm[pt] = point_position(<vec2d *> (targ.data), num_cams,
                                 cparam._control_par.mm, calib,
                                 <vec3d> np.PyArray_GETPTR2(res, pt, 0))

    return res, rcm
