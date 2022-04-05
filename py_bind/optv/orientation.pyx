import numpy as np
cimport numpy as np

ctypedef np.float64_t pos_t
from libc.stdlib cimport calloc, free

from optv.tracking_framebuf cimport TargetArray
from optv.calibration cimport Calibration
from optv.parameters cimport ControlParams, VolumeParams
from optv.epipolar cimport epi_mm_2D


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
        # raise ValueError('Must have at least as many targets as ref. points.')
        print('Must have at least as many targets as ref. points.')
        pass

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
                    ControlParams cparam, cals, VolumeParams vparam):
    """
    Calculate the 3D positions of the points given by their 2D projections
    using one of the options: 
    - for a single camera, uses single_cam_point_positions()
    - for multiple cameras, uses multi_cam_point_positions()

    Arguments:
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the
        camera order of ``targets``.
    VolumeParams vparam - an object holding observed volume size parameters, needed
        for the single camera case only.

    Returns:
    res - (n,3) array for n points represented by their targets.
    rcm - n-length array, the Ray Convergence Measure for eachpoint for multi camera
        option, or zeros for a single camera option
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] res
        np.ndarray[ndim=1, dtype=pos_t] rcm

    if len(cals) == 1:
        res, rcm = single_cam_point_positions(targets, cparam, cals, vparam)
    elif len(cals) > 1:
        res, rcm = multi_cam_point_positions(targets,cparam,cals)
    else:
        raise ValueError("wrong number of cameras in point_positions")
    
    return res, rcm


def single_cam_point_positions(np.ndarray[ndim=3, dtype=pos_t] targets,
                    ControlParams cparam, cals, VolumeParams vparam):
    """
    Calculates the 3D positions of the points from a single camera using
    the 2D target positions given in metric coordinates

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
    rcm = np.zeros(num_targets)

    for pt in range(num_targets):
        targ = targets[pt]
        epi_mm_2D (targ[0][0], targ[0][1],
                            calib[0], cparam._control_par.mm, vparam._volume_par, 
                            <vec3d> np.PyArray_GETPTR2(res, pt, 0));

    return res, rcm
          

def multi_cam_point_positions(np.ndarray[ndim=3, dtype=pos_t] targets,
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

def external_calibration(Calibration cal, 
    np.ndarray[ndim=2, dtype=pos_t] ref_pts, 
    np.ndarray[ndim=2, dtype=pos_t] img_pts,
    ControlParams cparam):
    """
    Update the external calibration with results of raw orientation, i.e.
    the iterative process that adjust the initial guess' external parameters
    (position and angle of cameras) without internal or distortions.
    
    Arguments:
    Calibration cal - position and other parameters of the camera.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known 
        positions of the select 2D points found on the image.
    np.ndarray[ndim=2, dtype=pos_t] img_pts - a selection of pixel coordinates
        of image points whose 3D position is known.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    True if iteration succeeded, false otherwise.
    """
    cdef:
        target *targs
        vec3d *ref_coord
    
    ref_pts = np.ascontiguousarray(ref_pts)
    ref_coord = <vec3d *>ref_pts.data
    
    # Convert pixel coords to metric coords:
    targs = <target *>calloc(len(img_pts), sizeof(target))
    
    for ptx, pt in enumerate(img_pts):
        targs[ptx].x = pt[0]
        targs[ptx].y = pt[1]
    
    success = raw_orient (cal._calibration, cparam._control_par, 
        len(ref_pts), ref_coord, targs)
    
    free(targs);
    
    return (True if success else False)

def full_calibration(Calibration cal,
    np.ndarray[ndim=2, dtype=pos_t] ref_pts, TargetArray img_pts,
    ControlParams cparam, list flags=[]):
    """
    Performs a full calibration, affecting all calibration structs.
    
    Arguments:
    Calibration cal - current position and other parameters of the camera. Will
        be overwritten with new calibration if iteration succeeded, otherwise 
        remains untouched.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known 
        positions of the select 2D points found on the image.
    TargetArray img_pts - detected points to match to known 3D positions.
        Must be sorted by matching ref point (as done by function
        ``match_detection_to_ref()``.
    ControlParams cparam - an object holding general control parameters.
    flags - a list whose members are the names of possible distortion
        parameters. Only parameter names present in the list will be used.
        Passing an empty list should be functionally equivalent to a raw 
        calibration, though the code paths taken in C are different. 
        
        The recognized flags are: 
            'cc', 'xh', 'yh' - sensor position.
            'k1', 'k2', 'k3' - radial distortion.
            'p1', 'p2' - decentering
            'scale', 'shear' - affine transforms.
        
        This is what the underlying library uses a struct for, but come on.
    
    Returns:
    ret - (r,2) array, the residuals in the x and y direction for r points used
        in orientation.
    used - r-length array, indices into target array of targets used.
    err_est - error estimation per calibration DOF. We 
    
    Raises:
    ValueError if iteration did not converge.
    """
    cdef:
        vec3d *ref_coord
        np.ndarray[ndim=2, dtype=pos_t] ret
        np.ndarray[ndim=1, dtype=np.int_t] used
        np.ndarray[ndim=1, dtype=pos_t] err_est
        orient_par *orip
        double *residuals
    
    ref_pts = np.ascontiguousarray(ref_pts)
    ref_coord = <vec3d *>ref_pts.data

    # Load up the orientation parameters. Silly, but saves on defining
    # a whole new class for what is no more than a list.
    orip = <orient_par *>calloc(1, sizeof(orient_par))
    orip[0].useflag = 0
    orip[0].ccflag = (1 if 'cc' in flags else 0)
    orip[0].xhflag = (1 if 'xh' in flags else 0)
    orip[0].yhflag = (1 if 'yh' in flags else 0)
    orip[0].k1flag = (1 if 'k1' in flags else 0)
    orip[0].k2flag = (1 if 'k2' in flags else 0)
    orip[0].k3flag = (1 if 'k3' in flags else 0)
    orip[0].p1flag = (1 if 'p1' in flags else 0)
    orip[0].p2flag = (1 if 'p2' in flags else 0)
    orip[0].scxflag = (1 if 'scale' in flags else 0)
    orip[0].sheflag = (1 if 'shear' in flags else 0)
    orip[0].interfflag = 0 # This also solves for the glass, I'm skipping it.
    
    err_est = np.empty((NPAR + 1) * sizeof(double))
    residuals = orient(cal._calibration, cparam._control_par, len(ref_pts), 
        ref_coord, img_pts._tarr, orip, <double *>err_est.data)
    
    free(orip)
    
    if residuals == NULL:
        free(residuals)
        raise ValueError("Orientation iteration failed, need better setup.")
    
    ret = np.empty((len(img_pts), 2))
    used = np.empty(len(img_pts), dtype=np.int_)
    
    for ix in range(len(img_pts)):
        ret[ix] = (residuals[2*ix], residuals[2*ix + 1])
        used[ix] = img_pts[ix].pnr()
    
    free(residuals)
    return ret, used, err_est

def dumbbell_target_func(np.ndarray[ndim=3, dtype=pos_t] targets, 
    ControlParams cparam, cals, db_length, db_weight):
    """
    Wrap the epipolar convergence test.
    
    Arguments:
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the 
        camera order of ``targets``.
    db_length - distance between two dumbbell targets.
    db_weight - weight of relative dumbbell size error in target function.
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] targ
        vec2d **ctargets
        calibration **calib = cal_list2arr(cals)
        int cam, num_cams
    
    num_cams = targets.shape[1]
    num_pts = targets.shape[0]
    ctargets = <vec2d **>calloc(num_pts, sizeof(vec2d*))
    
    for pt in range(num_pts):
        targ = targets[pt]
        ctargets[pt] = <vec2d *>(targ.data)
    
    return weighted_dumbbell_precision(ctargets, num_pts, num_cams, 
        cparam._control_par.mm, calib,  db_length, db_weight)
