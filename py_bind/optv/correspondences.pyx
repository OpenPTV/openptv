# -*- coding: utf-8 -*-
"""
Implementation of bindings for correspondences and related data structures.

Created on Fri Oct 28 13:46:39 2016

@author: yosef
"""

from libc.stdlib cimport malloc, calloc, free
cimport numpy as np
import numpy as np

from optv.transforms cimport pixel_to_metric, dist_to_flat
from optv.parameters cimport ControlParams, VolumeParams
from optv.calibration cimport Calibration
from optv.orientation cimport COORD_UNUSED
from optv.tracking_framebuf cimport TargetArray, Target, target, frame, \
    PT_UNUSED, CORRES_NONE

cdef class MatchedCoords:
    """
    Keeps a block of 2D flat coordinates, each with a "point number", the same
    as the number on one ``target`` from the block to which this block is kept
    matched. This block is x-sorted.
    
    NB: the data is not meant to be directly manipulated at this point. The 
    coord_2d arrays are most useful as intermediate objects created and 
    manipulated only by other liboptv functions. Although one can imagine a 
    use case for direct manipulation in Python, it is rare and supporting it 
    is a low priority.
    """
    
    def __init__(
        self, TargetArray targs, ControlParams cpar, 
        Calibration cal, double tol=0.00001, reset_numbers=True):
        """
        Allocates and initializes the memory, including coordinate conversion 
        and sorting.
        
        Arguments:
        TargetArray targs - the TargetArray to be converted and matched.
        ControlParams cpar - parameters of image size etc. for conversion.
        Calibration cal - representation of the camera parameters to use in
            the flat/distorted transforms.
        double tol - optional tolerance for the lens distortion correction 
            phase, see ``optv.transforms``.
        reset_numbers - if True (default) numbers the targets too, in their 
            current order. This shouldn't be necessary since all TargetArray
            creators number the targets, but this gets around cases where they
            don't.
        """
        cdef:
            target *targ
        
        self._num_pts = len(targs)
        self.buf = <coord_2d *> malloc(self._num_pts * sizeof(coord_2d))
        if self.buf == NULL:
            raise MemoryError("could not allocate matched-coordinates array.")
        
        for tnum in range(self._num_pts):
            targ = &(targs._tarr[tnum])
            if reset_numbers:
                targ.pnr = tnum
            
            pixel_to_metric(
                &(self.buf[tnum].x), &(self.buf[tnum].y), targ.x, targ.y, 
                cpar._control_par)
            dist_to_flat(
                self.buf[tnum].x, self.buf[tnum].y, cal._calibration,
                &(self.buf[tnum].x), &(self.buf[tnum].y), tol)
            self.buf[tnum].pnr = targ.pnr
        
        quicksort_coord2d_x(self.buf, self._num_pts)
    
    def as_arrays(self):
        """
        Returns the data associated with the object (the matched coordinates 
        block) as NumPy arrays.
        
        Returns:
        pos - (n,2) array, the (x,y) flat-coordinates position of n targets.
        pnr - n-length array, the corresponding target number for each point.
        """
        cdef:
            np.ndarray[ndim=2, dtype=np.float64_t] pos
            np.ndarray[ndim=1, dtype=np.int64_t] pnr
            int pt
        
        pos = np.empty((self._num_pts, 2))
        pnr = np.empty(self._num_pts, dtype=np.int_)
        
        for pt in range(self._num_pts):
            pos[pt,0] = self.buf[pt].x
            pos[pt,1] = self.buf[pt].y
            pnr[pt] = self.buf[pt].pnr
        
        return pos, pnr
    
    def get_by_pnrs(self, np.ndarray[ndim=1, dtype=np.int64_t] pnrs):
        """
        Return the flat positions of points whose pnr property is given, as an
        (n,2) flat position array. Assumes all pnrs are to be found, otherwise
        there will be garbage at the end of the position array.
        """
        cdef:
            np.ndarray[ndim=2, dtype=np.float64_t] pos
            int pt
        
        pos = np.full((len(pnrs), 2), COORD_UNUSED, dtype=np.float64)
        for pt in range(self._num_pts):
            which = np.flatnonzero(self.buf[pt].pnr == pnrs)
            if len(which) > 0:
                which = which[0]
                pos[which,0] = self.buf[pt].x
                pos[which,1] = self.buf[pt].y
        return pos
        
    def __dealloc__(self):
        free(self.buf)

def correspondences(list img_pts, list flat_coords, list cals, 
    VolumeParams vparam, ControlParams cparam):
    """
    Get the correspondences for each clique size. 
    
    Arguments:
    img_pts - a list of c := len(cals), containing TargetArray objects, each 
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property 
        set. the pnr property should be set to the target index in its array.
    flat_coords - a list of MatchedCoordinates objects, one per camera, holding
        the x-sorted flat-coordinates conversion of the respective image 
        targets.
    cals - a list of Calibration objects, each for the camera taking one image.
    VolumeParams vparam - an object holding observed volume size parameters.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of 
        c image planes of points belonging to quadruplets, triplets, pairs 
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belonging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of 
        previous 3).
    """
    cdef: 
        int num_cams = len(cals)

    # Special case of a single camera, follow the single_cam_correspondence docstring    
    if num_cams == 1:
        sorted_pos, sorted_corresp, num_targs = single_cam_correspondence(img_pts, flat_coords, cals)
        return sorted_pos, sorted_corresp, num_targs

    cdef:        
        calibration **calib = <calibration **> malloc(
            num_cams * sizeof(calibration *))
        coord_2d **corrected = <coord_2d **> malloc(
            num_cams * sizeof(coord_2d *))
        frame frm
        
        np.ndarray[ndim=2, dtype=np.int64_t] clique_ids
        np.ndarray[ndim=3, dtype=np.float64_t] clique_targs
        
        # Return buffers:
        int *match_counts = <int *> malloc(num_cams * sizeof(int))
        n_tupel *corresp_buf
    
    # Initialize frame partially, without the extra momory used by init_frame.
    frm.targets = <target**> calloc(num_cams, sizeof(target*))
    frm.num_targets = <int *> calloc(num_cams, sizeof(int))
    
    for cam in range(num_cams):
        calib[cam] = (<Calibration>cals[cam])._calibration
        frm.targets[cam] = (<TargetArray>img_pts[cam])._tarr
        frm.num_targets[cam] = len(img_pts[cam])
        corrected[cam] = (<MatchedCoords>flat_coords[cam]).buf
        
    # The biz:
    corresp_buf = corresp(&frm, corrected, 
        vparam._volume_par, cparam._control_par, calib, match_counts)
    
    # Distribute data to return structures:
    sorted_pos = [None]*(num_cams - 1)
    sorted_corresp = [None]*(num_cams - 1)
    last_count = 0
    
    for clique_type in range(num_cams - 1): 
        num_points = match_counts[4 - num_cams + clique_type] # for 1-4 cameras
        clique_targs = np.full((num_cams, num_points, 2), PT_UNUSED, 
            dtype=np.float64)
        clique_ids = np.full((num_cams, num_points), CORRES_NONE, 
            dtype=np.int_)
        
        # Trace back the pixel target properties through the flat metric
        # intermediary that's x-sorted.
        for cam in range(num_cams):            
            for pt in range(num_points):
                geo_id = corresp_buf[pt + last_count].p[cam]
                if geo_id < 0:
                    continue
                
                p1 = corrected[cam][geo_id].pnr
                clique_ids[cam, pt] = p1

                if p1 > -1:
                    targ = img_pts[cam][p1]
                    clique_targs[cam, pt, 0] = (<Target> targ)._targ.x
                    clique_targs[cam, pt, 1] = (<Target> targ)._targ.y
        
        last_count += num_points
        sorted_pos[clique_type] = clique_targs
        sorted_corresp[clique_type] = clique_ids
    
    # Clean up.
    num_targs = match_counts[num_cams - 1]
    free(frm.targets)
    free(frm.num_targets)
    free(calib)
    free(match_counts)
    free(corresp_buf) # Note this for future returning of correspondences.
    
    return sorted_pos, sorted_corresp, num_targs

def single_cam_correspondence(list img_pts, list flat_coords, list cals):
    """ 
    Single camera correspondence is not a real correspondence, it will be only a projection 
    of a 2D target from the image space into the 3D position, x,y,z using epi_mm_2d 
    function. Here we only update the pointers of the targets and return it in a proper format. 
    
     Arguments:
    img_pts - a list of c := len(cals), containing TargetArray objects, each 
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property 
        set. the pnr property should be set to the target index in its array.
    flat_coords - a list of MatchedCoordinates objects, one per camera, holding
        the x-sorted flat-coordinates conversion of the respective image 
        targets.
    cals - a list of Calibration objects, each for the camera taking one image.

    Returns:
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of 
        c image planes of points belonging to quadruplets, triplets, pairs 
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belonging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of 
        previous 3). 
    """
    cdef: 
        int pt, num_points
        coord_2d *corrected = <coord_2d *> malloc(sizeof(coord_2d *))
    
    corrected = (<MatchedCoords>flat_coords[0]).buf

    sorted_pos = [None]
    sorted_corresp = [None]

    num_points = len(img_pts[0])

    clique_targs = np.full((1, num_points, 2), PT_UNUSED, 
        dtype=np.float64)
    clique_ids = np.full((1, num_points), CORRES_NONE, 
        dtype=np.int_)

    # Trace back the pixel target properties through the flat metric
    # intermediary that's x-sorted.
    for pt in range(num_points):

        # From Beat code (issue #118) pix[0][geo[0][i].pnr].tnr=i;

        p1 = corrected[pt].pnr
        clique_ids[0, pt] = p1

        if p1 > -1:
            targ = img_pts[0][p1]
            clique_targs[0, pt, 0] = (<Target> targ)._targ.x
            clique_targs[0, pt, 1] = (<Target> targ)._targ.x
            # we also update the tnr, see docstring of correspondences
            (<Target> targ)._targ.tnr = pt

    sorted_pos[0] = clique_targs
    sorted_corresp[0] = clique_ids

    return sorted_pos, sorted_corresp, num_points
