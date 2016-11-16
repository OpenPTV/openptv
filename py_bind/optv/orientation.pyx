import numpy as np
cimport numpy as np

ctypedef np.float64_t pos_t
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

    if len(img_pts) != len(ref_pts):
        raise TypeError('Lengths of ref_pts and img_pts must be equal.')

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
