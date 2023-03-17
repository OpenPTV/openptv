"""
    /*******************************************************************

   Routine:	        track.c

   Author/Copyright:        Jochen Willneff

   Address:	        Institute of Geodesy and Photogrammetry
                    ETH - Hoenggerberg
                    CH - 8093 Zurich

   Creation Date:		Beginning: February '98
                        End: far away

   Description:             Tracking of particles in image- and objectspace

   Routines contained:      trackcorr_c

   Updated:           Yosef Meller and Alex Liberzon
   Address:           Tel Aviv University
   For:               OpenPTV, http://www.openptv.net
   Modification date: October 2016

*******************************************************************/

/* References:
   [1] http://en.wikipedia.org/wiki/Gradian
 */
    
"""


import math
    
# Definitions for tracking routines.

from tracking_frame_buf import *
from parameters import *
from trafo import *
from tracking_run import *
from vec_utils import *
from imgcoord import *
from multimed import *
from orientation import *
from calibration import *


TR_UNUSED = -1
TR_BUFSPACE = 4
TR_MAX_CAMS = 4
MAX_TARGETS = 20000
MAX_CANDS = 4
ADD_PART = 3

class Foundpix:
    def __init__(self):
        self.ftnr = 0
        self.freq = 0
        self.whichcam = [0] * TR_MAX_CAMS
        


from typing import List


from typing import List

class tracking_run:
    def __init__(self):
        self.fb = None
        self.seq_par = None
        self.tpar = None
        self.vpar = None
        self.cpar = None
        self.cal = None
        self.flatten_tol = None
        self.ymin = None
        self.ymax = None
        self.lmax = None
        self.npart = None
        self.nlinks = None

def tr_new_legacy(seq_par_fname: str, tpar_fname: str, vpar_fname: str,
                   cpar_fnamei: str, cal: List[Calibration]) -> tracking_run:
    # implementation not provided
    pass

def tr_new(seq_par: sequence_par, tpar: track_par, vpar: volume_par,
           cpar: control_par, buf_len: int, max_targets: int,
           corres_file_base: str, linkage_file_base: str, prio_file_base: str,
           cal: List[Calibration], flatten_tol: float) -> tracking_run:
    tr = tracking_run()
    tr.fb = framebuf_base(buf_len, max_targets, corres_file_base, linkage_file_base, prio_file_base)
    tr.seq_par = seq_par
    tr.tpar = tpar
    tr.vpar = vpar
    tr.cpar = cpar
    tr.cal = cal
    tr.flatten_tol = flatten_tol
    return tr

def tr_free(tr: tracking_run) -> None:
    tr.fb.free()
    del tr.seq_par
    del tr.tpar
    del tr.vpar
    del tr.cpar
    del tr.cal
    del tr
    


# The buffer space required for this algorithm:

# Note that MAX_TARGETS is taken from the global M, but I want a separate
# definition because the fb created here should be local, not used outside
# this file.

# MAX_CANDS is the max number of candidates sought in search volume for next
# link.

TR_BUFSPACE = 4
TR_MAX_CAMS = 4
MAX_TARGETS = 20000
MAX_CANDS = 4         # max candidates, nearest neighbours
ADD_PART = 3          # search region 3 pix around a particle

class FoundPix:
    def __init__(self):
        self.ftnr = 0
        self.freq = 0
        self.whichcam = [0]*TR_MAX_CAMS

def candsearch_in_pix(next, num_targets, x, y, dl, dr, du, dd, p, cpar):
    raise NotImplementedError

def candsearch_in_pix_rest(next, num_targets, x, y, dl, dr, du, dd, p, cpar):
    raise NotImplementedError

def sort_candidates_by_freq(item, num_cams):
    raise NotImplementedError

def searchquader(point, xr, xl, yd, yu, tpar, cpar, cal):
    raise NotImplementedError

def predict(a, b, c):
    raise NotImplementedError

def search_volume_center_moving(prev_pos, curr_pos, output):
    raise NotImplementedError

def pos3d_in_bounds(pos, bounds):
    raise NotImplementedError

def det_lsq_3d(cals, mm, v, Xp, Yp, Zp, num_cams):
    raise NotImplementedError

def sort(n, a, b):
    raise NotImplementedError

def angle_acc(start, pred, cand, angle, acc):
    raise NotImplementedError

def reset_foundpix_array(arr, arr_len, num_cams):
    raise NotImplementedError

def copy_foundpix_array(dest, src, arr_len, num_cams):
    raise NotImplementedError

def point_to_pixel(v1, point, cal, cpar):
    raise NotImplementedError

def track_forward_start(tr):
    raise NotImplementedError

def trackcorr_c_loop(run_info, step):
    raise NotImplementedError

def trackcorr_c_finish(run_info, step):
    raise NotImplementedError

def trackback_c(run_info):
    raise NotImplementedError



def track_forward_start(tr: tracking_run):
    """
    Initializes the tracking frame buffer with the first frames.
    Arguments:
    tr (tracking_run): An object holding the per-run tracking parameters and a frame buffer with 4 positions.
    """
    # Prime the buffer with first frames
    for step in range(tr.seq_par.first, tr.seq_par.first + TR_BUFSPACE - 1):
        fb_read_frame_at_end(tr.fb, step, 0)
        fb_next(tr.fb)
    fb_prev(tr.fb)
    

def reset_foundpix_array(arr, arr_len, num_cams):
    """
    reset_foundpix_array() sets default values for foundpix objects in an array.

    Arguments:
    arr -- the array to reset
    arr_len -- array length
    num_cams -- number of places in the whichcam member of foundpix.
    """
    for i in range(arr_len):
        # Set default values for each foundpix object in the array
        arr[i].ftnr = TR_UNUSED
        arr[i].freq = 0
        
        # Set default values for each whichcam member of the foundpix object
        for cam in range(num_cams):
            arr[i].whichcam[cam] = 0


def copy_foundpix_array(dest, src, arr_len, num_cams):
    """
    copy_foundpix_array() copies foundpix objects from one array to another.

    Arguments:
    dest -- dest receives the copied array
    src -- src is the array to copy
    arr_len -- array length
    num_cams -- number of places in the whichcam member of foundpix.
    """
    for i in range(arr_len):
        # Copy values from source foundpix object to destination foundpix object
        dest[i].ftnr = src[i].ftnr
        dest[i].freq = src[i].freq
        
        # Copy values from source whichcam member to destination whichcam member
        for cam in range(num_cams):
            dest[i].whichcam[cam] = src[i].whichcam[cam]


def register_closest_neighbs(targets, num_targets, cam, cent_x, cent_y, dl, dr, du, dd, reg, cpar):
    """
    register_closest_neighbs() finds candidates for continuing a particle's
    path in the search volume, and registers their data in a foundpix array
    that is later used by the tracking algorithm.

    Arguments:
    targets -- the targets list to search.
    num_targets -- target array length.
    cam -- the index of the camera we're working on.
    cent_x -- image coordinate of search area center along x-axis, [pixel]
    cent_y -- image coordinate of search area center along y-axis, [pixel]
    dl -- left distance to the search area border from its center, [pixel]
    dr -- right distance to the search area border from its center, [pixel]
    du -- up distance to the search area border from its center, [pixel]
    dd -- down distance to the search area border from its center, [pixel]
    reg -- an array of foundpix objects, one for each possible neighbour. Output array.
    cpar -- control parameter object
    """
    all_cands = [-999] * MAX_CANDS  # Initialize all candidate indexes to -999

    cand = candsearch_in_pix(targets, num_targets, cent_x, cent_y, dl, dr, du, dd, all_cands, cpar)

    for cand_idx in range(MAX_CANDS):
        # Set default value for unused foundpix objects
        if all_cands[cand_idx] == -999:
            reg[cand_idx].ftnr = TR_UNUSED
        else:
            # Register candidate data in the foundpix object
            reg[cand_idx].whichcam[cam] = 1
            reg[cand_idx].ftnr = targets[all_cands[cand_idx]].tnr
            

def search_volume_center_moving(prev_pos, curr_pos, output):
    """
    Finds the position of the center of the search volume for a moving particle using the velocity of last step.
    Args:
        prev_pos (vec3d): Previous position of the particle.
        curr_pos (vec3d): Current position of the particle.
        output (vec3d): Output variable for the calculated position.

    Returns:
        None
    """
    # Multiply current position by 2 and subtract previous position
    output[0] = 2 * curr_pos[0] - prev_pos[0]
    output[1] = 2 * curr_pos[1] - prev_pos[1]
    output[2] = 2 * curr_pos[2] - prev_pos[2]


def predict(prev_pos, curr_pos, output):
    """
    Predicts the position of a particle in the next frame, using the previous and current positions.
    Args:
        prev_pos (vec2d): 2D position at previous frame.
        curr_pos (vec2d): 2D position at current frame.
        output (vec2d): Output of the 2D positions of the particle in the next frame.

    Returns:
        None
    """
    # Calculate the position of the particle in the next frame using the current and previous positions
    output[0] = 2 * curr_pos[0] - prev_pos[0]
    output[1] = 2 * curr_pos[1] - prev_pos[1]


def pos3d_in_bounds(pos, bounds):
    """
    Checks that all components of a pos3d are in their respective bounds taken from a track_par object.
    Args:
        pos (vec3d): The 3-component array to check.
        bounds (track_par): The struct containing the bounds specification.

    Returns:
        True if all components are in bounds, False otherwise.
    """
    # Check if all three components of pos are within their respective bounds in bounds.
    return (
        bounds.dvxmin < pos[0] < bounds.dvxmax and
        bounds.dvymin < pos[1] < bounds.dvymax and
        bounds.dvzmin < pos[2] < bounds.dvzmax
    )


def angle_acc(start, pred, cand):
    """
    Calculates the angle between the (1st order) numerical velocity vectors to the predicted next position and to the candidate actual position.
    The angle is calculated in [gon], see [1].
    The predicted position is the position if the particle continued at current velocity.

    Arguments:
    start -- vec3d, the particle start position
    pred -- vec3d, predicted position
    cand -- vec3d, possible actual position

    Returns:
    angle -- float, the angle between the two velocity vectors, [gon]
    acc -- float, the 1st-order numerical acceleration embodied in the deviation from prediction.
    """
    v0 = [pred[i] - start[i] for i in range(3)]
    v1 = [cand[i] - start[i] for i in range(3)]

    acc = math.dist(v0, v1)

    if (v0[0] == -v1[0]) and (v0[1] == -v1[1]) and (v0[2] == -v1[2]):
        angle = 200
    elif (v0[0] == v1[0]) and (v0[1] == v1[1]) and (v0[2] == v1[2]):
        angle = 0  # otherwise it returns NaN
    else:
        angle = (200./math.pi) * math.acos(sum([v0[i] * v1[i] for i in range(3)]) / (math.dist(start, pred) * math.dist(start, cand)))

    return angle, acc


from math import sqrt

def candsearch_in_pix(next, num_targets, cent_x, cent_y, dl, dr, du, dd, cpar):
    p = [-1] * 4
    counter = 0
    dmin = 1e20
    p1 = p2 = p3 = p4 = -1
    d1 = d2 = d3 = d4 = dmin

    xmin, xmax, ymin, ymax = cent_x - dl, cent_x + dr, cent_y - du, cent_y + dd

    if xmin < 0:
        xmin = 0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0:
        ymin = 0
    if ymax > cpar.imy:
        ymax = cpar.imy

    if cent_x >= 0 and cent_x <= cpar.imx and cent_y >= 0 and cent_y <= cpar.imy:
        j0 = num_targets // 2
        dj = num_targets // 4
        while dj > 1:
            if next[j0].y < ymin:
                j0 += dj
            else:
                j0 -= dj
            dj //= 2

        j0 -= 12
        if j0 < 0:
            j0 = 0

        for j in range(j0, num_targets):
            if next[j].tnr != -1:
                if next[j].y > ymax:
                    break
                if xmin < next[j].x < xmax and ymin < next[j].y < ymax:
                    d = sqrt((cent_x - next[j].x) ** 2 + (cent_y - next[j].y) ** 2)

                    if d < dmin:
                        dmin = d

                    if d < d1:
                        p4, p3, p2, p1 = p3, p2, p1, j
                        d4, d3, d2, d1 = d3, d2, d1, d
                    elif d1 < d < d2:
                        p4, p3, p2 = p3, p2, j
                        d4, d3, d2 = d3, d2, d
                    elif d2 < d < d3:
                        p4, p3 = p3, j
                        d4, d3 = d3, d
                    elif d3 < d < d4:
                        p4 = j
                        d4 = d

        p[0] = p1
        p[1] = p2
        p[2] = p3
        p[3] = p4

        for j in range(4):
            if p[j] != -1:
                counter += 1

    return counter


import numpy as np

def candsearch_in_pix_rest(next, cent_x, cent_y, dl, dr, du, dd, cpar):
    """
    Searches for a nearest candidate in unmatched target list

    Arguments:
    next - 2D numpy array of targets (pointer, x,y, n, nx,ny, sumg, track ID), assumed to be y sorted.
    cent_x, cent_y - image coordinates of the position of a particle [pixel]
    dl, dr, du, dd - respectively the left, right, up, down distance to the search area borders from its center, [pixel]
    cpar - control_par object with attributes imx and imy.

    Returns:
    int - the number of candidates found, between 0 - 1
    """
    counter = 0
    dmin = 1e20
    xmin, xmax, ymin, ymax = cent_x - dl, cent_x + dr, cent_y - du, cent_y + dd

    xmin = max(xmin, 0.0)
    xmax = min(xmax, cpar.imx)
    ymin = max(ymin, 0.0)
    ymax = min(ymax, cpar.imy)

    p = np.array([-1], dtype=np.int32)

    if 0 <= cent_x <= cpar.imx and 0 <= cent_y <= cpar.imy:
        # binarized search for start point of candidate search
        j0, dj = next.shape[0] // 2, next.shape[0] // 4
        while dj > 1:
            j0 += dj if next[j0, 1] < ymin else -dj
            dj //= 2

        j0 -= 12 if j0 >= 12 else j0  # due to trunc
        for j in range(j0, next.shape[0]):
            if next[j, 3] == -1:  # tnr == TR_UNUSED
                if next[j, 1] > ymax:
                    break  # finish search
                if xmin < next[j, 0] < xmax and ymin < next[j, 1] < ymax:
                    d = np.sqrt((cent_x - next[j, 0]) ** 2 + (cent_y - next[j, 1]) ** 2)
                    if d < dmin:
                        dmin = d
                        p[0] = j

        if p[0] != -1:
            counter += 1

    return counter, p

import numpy as np

def searchquader(point, tpar, cpar, cal):
    def vec_set(vec, x, y, z):
        vec[0], vec[1], vec[2] = x, y, z

    def vec_copy(dest, src):
        dest[0], dest[1], dest[2] = src[0], src[1], src[2]

    def point_to_pixel(pixel, point, cal, cpar):
        pixel[0] = cal.camplane_u(cpar, point)
        pixel[1] = cal.camplane_v(cpar, point)

    def project_to_pixel(corners, point, mins, maxes, cal, cpar):
        for pt in range(8):
            vec_copy(corners[pt], point)
            for dim in range(3):
                if pt & 1 << dim:
                    corners[pt][dim] += maxes[dim]
                else:
                    corners[pt][dim] += mins[dim]

            point_to_pixel(corners[pt], corners[pt], cal, cpar)

    xr, xl, yd, yu = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mins, maxes = np.zeros(3), np.zeros(3)
    quader = np.zeros((8, 3))
    center = np.zeros(2)
    corner = np.zeros(2)

    vec_set(mins, tpar.dvxmin, tpar.dvymin, tpar.dvzmin)
    vec_set(maxes, tpar.dvxmax, tpar.dvymax, tpar.dvzmax)

    # 3D positions of search volume - eight corners of a box
    project_to_pixel(quader, point, mins, maxes, cal[0], cpar)
    
    # calculation of search area in each camera
    for i in range(cpar.num_cams):
        # initially large or small values
        xr[i] = 0
        xl[i] = cpar.imx
        yd[i] = 0
        yu[i] = cpar.imy

        # pixel position of a search center
        point_to_pixel(center, point, cal[i], cpar)

        # mark 4 corners of the search region in pixels
        for pt in range(8):
            point_to_pixel(corner, quader[pt], cal[i], cpar)

            if corner[0] < xl[i]:
                xl[i] = corner[0]
            if corner[1] < yu[i]:
                yu[i] = corner[1]
            if corner[0] > xr[i]:
                xr[i] = corner[0]
            if corner[1] > yd[i]:
                yd[i] = corner[1]

        if xl[i] < 0:
            xl[i] = 0
        if yu[i] < 0:
            yu[i] = 0
        if xr[i] > cpar.imx:
            xr[i] = cpar.imx
        if yd[i] > cpar.imy:
            yd[i] = cpar.imy

        # eventually xr,xl,yd,yu are pixel distances relative to the point
        xr[i] = xr[i] - center[0]
        xl[i] = center[0] - xl[i]
        yd[i] = yd[i] - center[1]
        yu[i] = center[1] - yu[i]


def sort_candidates_by_freq(item, num_cams):
    class FoundPix:
        def __init__(self, ftnr=0, whichcam=[0]*4, freq=0):
            self.ftnr = ftnr
            self.whichcam = whichcam
            self.freq = freq

    MAX_CANDS = 1000
    foundpix = [FoundPix() for i in range(num_cams*MAX_CANDS)]
    foundpix[:len(item)] = item
    
    different = 0

    # where what was found
    for i in range(num_cams*MAX_CANDS):
        for j in range(num_cams):
            for m in range(MAX_CANDS):
                if foundpix[i].ftnr == foundpix[4*j+m].ftnr:
                    foundpix[i].whichcam[j] = 1

    # how often was ftnr found
    for i in range(num_cams*MAX_CANDS):
        for j in range(num_cams):
            if foundpix[i].whichcam[j] == 1 and foundpix[i].ftnr != TR_UNUSED:
                foundpix[i].freq += 1

    # sort freq
    for i in range(1, num_cams*MAX_CANDS):
        for j in range(num_cams*MAX_CANDS-1, i-1, -1):
            if foundpix[j-1].freq < foundpix[j].freq:
                foundpix[j-1], foundpix[j] = foundpix[j], foundpix[j-1]

    # prune the duplicates or those that are found only once
    for i in range(num_cams*MAX_CANDS):
        for j in range(i+1, num_cams*MAX_CANDS):
            if foundpix[i].ftnr == foundpix[j].ftnr or foundpix[j].freq < 2:
                foundpix[j].freq = 0
                foundpix[j].ftnr = TR_UNUSED

    # sort freq again on the clean dataset
    for i in range(1, num_cams*MAX_CANDS):
        for j in range(num_cams*MAX_CANDS-1, i-1, -1):
            if foundpix[j-1].freq < foundpix[j].freq:
                foundpix[j-1], foundpix[j] = foundpix[j], foundpix[j-1]

    for i in range(num_cams*MAX_CANDS):
        if foundpix[i].freq != 0:
            different += 1

    return different

import numpy as np

def sort(a, b):
    """
    Sorts a float array 'a' and an integer array 'b' both of length n.
    
    Arguments:
    a -- float array (returned sorted in ascending order)
    b -- integer array (returned sorted according to float array a)
    
    Returns:
    Sorted arrays a and b.
    """
    n = len(a)
    idx = np.argsort(a)
    a = a[idx]
    b = b[idx]
    return a, b


import numpy as np

def point_to_pixel(point, cal, cpar):
    """
    Returns vec2d with pixel positions (x,y) in the camera.
    
    Arguments:
    point -- vec3d point in 3D space
    cal -- Calibration parameters
    cpar -- Control parameters (num cams, multimedia parameters, cpar->mm, etc.)
    
    Returns:
    vec2d with pixel positions (x,y) in the camera.
    """
    
    x, y = img_coord(point, cal, cpar.mm)
    x, y = metric_to_pixel(x, y, cpar)
    return np.array([x, y])



def sorted_candidates_in_volume(center, center_proj, frm, run):
    points = []
    right, left, down, up = [0]*TR_MAX_CAMS, [0]*TR_MAX_CAMS, [0]*TR_MAX_CAMS, [0]*TR_MAX_CAMS
    num_cams = frm.num_cams
    
    points = [foundpix() for _ in range(num_cams*MAX_CANDS)]
    reset_foundpix_array(points, num_cams*MAX_CANDS, num_cams)
    
    # Search limits in image space
    searchquader(center, right, left, down, up, run.tpar, run.cpar, run.cal)
    
    # search in pix for candidates in the next time step
    for cam in range(num_cams):
        register_closest_neighbs(
            frm.targets[cam], frm.num_targets[cam], cam, center_proj[cam][0], center_proj[cam][1],
            left[cam], right[cam], up[cam], down[cam], points[cam*MAX_CANDS], run.cpar
        )

    # fill and sort candidate struct
    num_cands = sort_candidates_by_freq(points, num_cams)
    if num_cands > 0:
        points = points[:num_cands] + [foundpix(ftnr=TR_UNUSED)]
        return points
    else:
        return None


def assess_new_position(pos, targ_pos, cand_inds, frm, run):
    """
    Determines the nearest target on each camera around a search position and 
    prepares the data structures accordingly with the determined target info or 
    the unused flag value.

    Arguments:
    pos - vec3d, the position around which to search.
    targ_pos - vec2d, the determined targets' respective positions.
    cand_inds - 2D array of integers, output buffer, the determined targets' 
        index in the respective camera's target list.
    frm - frame object, holdin target data for the search position.
    run - tracking_run object, scene information struct.

    Returns:
    Integer, the number of cameras where a suitable target was found.
    """

    # Initialize variables
    num_cands = 0
    valid_cams = 0
    _ix = 0
    pixel = [0, 0]
    left = right = up = down = ADD_PART

    for cam in range(TR_MAX_CAMS):
        targ_pos[cam][0] = targ_pos[cam][1] = COORD_UNUSED

    # Loop through cameras to find the nearest target
    for cam in range(run.cpar.num_cams):
        # Convert 3D point to 2D pixel coordinates
        point_to_pixel(pixel, pos, run.cal[cam], run.cpar)

        # Search for nearest target in pixel coordinates
        num_cands = candsearch_in_pix_rest(frm.targets[cam], frm.num_targets[cam],
                                           pixel[0], pixel[1], left, right, up, down,
                                           cand_inds[cam], run.cpar)

        if num_cands > 0:
            _ix = cand_inds[cam][0]  # first nearest neighbour
            targ_pos[cam][0] = frm.targets[cam][_ix].x
            targ_pos[cam][1] = frm.targets[cam][_ix].y

    # Loop through cameras to check if the target was found and calculate
    # target positions in metric coordinates
    for cam in range(run.cpar.num_cams):
        if targ_pos[cam][0] != COORD_UNUSED and targ_pos[cam][1] != COORD_UNUSED:
            pixel_to_metric(targ_pos[cam], targ_pos[cam], targ_pos[cam][0], targ_pos[cam][1], run.cpar)
            dist_to_flat(targ_pos[cam][0], targ_pos[cam][1], run.cal[cam],
                         targ_pos[cam], targ_pos[cam]+1, run.flatten_tol)
            valid_cams += 1

    return valid_cams


def add_particle(frm, pos, cand_inds):
    num_parts = frm.num_parts
    ref_path_inf = frm.path_info[num_parts]
    vec_copy(ref_path_inf.x, pos)
    reset_links(ref_path_inf)

    ref_corres = frm.correspond[num_parts]
    ref_targets = frm.targets
    for cam in range(frm.num_cams):
        ref_corres.p[cam] = CORRES_NONE

        # We always take the 1st candidate, apparently. Why did we fetch 4?
        if cand_inds[cam][0] != PT_UNUSED:
            _ix = cand_inds[cam][0]
            ref_targets[cam][_ix].tnr = num_parts
            ref_corres.p[cam] = _ix
            ref_corres.nr = num_parts

    frm.num_parts += 1



def trackcorr_c_loop(run_info, step):
    # sequence loop
    j, h, mm, kk, in_volume = 0, 0, 0, 0, 0
    philf = [[0 for _ in range(MAX_CANDS)] for _ in range(4)]
    count1, count2, count3, num_added = 0, 0, 0, 0
    quali = 0
    diff_pos, X = vec3d(), [vec3d() for _ in range(6)]   # 7 reference points used in the algorithm, TODO: check if can reuse some
    angle, acc, angle0, acc0, dl = 0.0, 0.0, 0.0, 0.0, 0.0
    angle1, acc1 = 0.0, 0.0
    v1, v2 = [vec2d() for _ in range(4)], [vec2d() for _ in range(4)]   # volume center projection on cameras
    rr = 0.0

    # Shortcuts to inside current frame
    curr_path_inf, ref_path_inf = None, None
    curr_corres = None
    curr_targets = None
    _ix = 0    # For use in any of the complex index expressions below
    orig_parts = 0    # avoid infinite loop with particle addition set

    # Shortcuts into the tracking_run struct
    cal = None
    fb = None
    tpar = None
    vpar = None
    cpar = None

    w, wn = None, None
    count1, num_added = 0, 0

    fb = run_info.fb
    cal = run_info.cal
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    curr_targets = fb.buf[1].targets

    # try to track correspondences from previous 0 - corp, variable h
    orig_parts = fb.buf[1].num_parts
    for h in range(orig_parts):
        for j in range(6):
            vec_init(X[j])

        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres = fb.buf[1].correspond[h]

        curr_path_inf.inlist = 0

        # 3D-position
        vec_copy(X[1], curr_path_inf.x)

        # use information from previous to locate new search position
        # and to calculate values for search area
        if curr_path_inf.prev >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev]
            vec_copy(X[0], ref_path_inf.x)
            search_volume_center_moving(ref_path_inf.x, curr_path_inf.x, X[2])

            for j in range(fb.num_cams):
                point_to_pixel(v1[j], X[2], cal[j], cpar)
        else:
            vec_copy(X[2], X[1])
            for j in range(fb.num_cams):
                if curr_corres.p[j] == CORRES_NONE:
                    point_to_pixel(v1[j], X[2], cal[j], cpar)
                else:
                    _ix = curr_corres.p[j]
                    v1[j][0] = curr_targets[j][_ix].x
                    v1[j][1] = curr_targets[j][_ix].y


        # calculate search cuboid and reproject it to the image space
        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2], run_info)
        if w is None:
            continue

        # Continue to find candidates for the candidates.
        count2 += 1
        mm = 0
        while w[mm].ftnr != TR_UNUSED:       # counter1-loop
            # search for found corr of current the corr in next with predicted location

            # found 3D-position
            ref_path_inf = fb.buf[2].path_info[w[mm].ftnr]
            vec_copy(X[3], ref_path_inf.x)

            if curr_path_inf.prev >= 0:
                for j in range(3):
                    X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])
            else:
                search_volume_center_moving(X[1], X[3], X[5])

            for j in range(fb.num_cams):
                point_to_pixel(v1[j], X[5], cal[j], cpar)

            # end of search in pix
            wn = sorted_candidates_in_volume(X[5], v1, fb.buf[3], run_info)
            if wn is not None:
                count3 += 1
                kk = 0
                while wn[kk].ftnr != TR_UNUSED:
                    ref_path_inf = fb.buf[3].path_info[wn[kk].ftnr]
                    vec_copy(X[4], ref_path_inf.x)

                    vec_subt(X[4], X[3], diff_pos)
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle_acc(X[3], X[4], X[5], angle1, acc1)
                        if curr_path_inf.prev >= 0:
                            angle_acc(X[1], X[2], X[3], angle0, acc0)
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) / 2
                        angle = (angle0 + angle1) / 2
                        quali = wn[kk].freq + w[mm].freq

                        if acc < tpar.dacc and angle < tpar.dangle or acc < tpar.dacc / 10:
                            dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[mm].ftnr)

                    kk += 1  # End of searching 2nd-frame candidates.

        # creating new particle position,
        # reset img coord because of num_cams < 4
        # fix distance of 3 pixels to define xl,xr,yu,yd instead of searchquader
        # and search for unused candidates in next time step

        quali = assess_new_position(X[5], v2, philf, fb.buf[3], run_info)

        # quali >=2 means at least in two cameras
        # we found a candidate
        if quali >= 2:
            in_volume = 0                 # inside volume

            dl = point_position(v2, cpar.num_cams, cpar.mm, cal, X[4])

            # volume check
            if vpar.X_lay[0] < X[4][0] and X[4][0] < vpar.X_lay[1] \
                and run_info.ymin < X[4][1] and X[4][1] < run_info.ymax \
                and vpar.Zmin_lay[0] < X[4][2] and X[4][2] < vpar.Zmax_lay[1]:
                in_volume = 1

            vec_subt(X[3], X[4], diff_pos)
            if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                angle_acc(X[3], X[4], X[5], angle, acc)

                if acc < tpar.dacc and angle < tpar.dangle \
                    or acc < tpar.dacc/10:
                    dl = (vec_diff_norm(X[1], X[3]) +
                        vec_diff_norm(X[4], X[3])) / 2
                    rr = (dl/run_info.lmax + acc/tpar.dacc + angle/tpar.dangle) / \
                        (quali + w[mm].freq)
                    register_link_candidate(curr_path_inf, rr, w[mm].ftnr)

                    if tpar.add:
                        add_particle(fb.buf[3], X[4], philf)
                        num_added += 1

            in_volume = 0
        quali = 0

        # end of creating new particle position
        # ***************************************************************

        # try to link if kk is not found/good enough and prev exist
        if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
            diff_pos = vec_subt(X[3], X[1])
            if pos3d_in_bounds(diff_pos, tpar):
                angle, acc = angle_acc(X[1], X[2], X[3])
                if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                    quali = w[mm].freq
                    dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                    rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                    register_link_candidate(curr_path_inf, rr, w[mm].ftnr)

            mm += 1  # increment mm

        # begin of inlist still zero
        if tpar.add:
            if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
                quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info)
                if quali >= 2:
                    X[3] = vec_copy(X[2])
                    in_volume = 0
                    dl = point_position(v2, fb.num_cams, cpar.mm, cal, X[3])

                    # in volume check
                    if vpar.X_lay[0] < X[3][0] < vpar.X_lay[1] and run_info.ymin < X[3][1] < run_info.ymax and \
                            vpar.Zmin_lay[0] < X[3][2] < vpar.Zmax_lay[1]:
                        in_volume = 1

                    diff_pos = vec_subt(X[2], X[3])
                    if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                            dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, fb.buf[2].num_parts)
                            add_particle(fb.buf[2], X[3], philf)
                            num_added += 1
                    in_volume = 0

        # end of inlist still zero
        # ***********************************

        free(w)


    # sort decis and give preliminary "finaldecis"
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]

        if curr_path_inf.inlist > 0:
            sort(curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis)
            curr_path_inf.finaldecis = curr_path_inf.decis[0]
            curr_path_inf.next = curr_path_inf.linkdecis[0]

    # create links with decision check
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]

        if curr_path_inf.inlist > 0:
            ref_path_inf = fb.buf[2].path_info[curr_path_inf.next]

            if ref_path_inf.prev == PREV_NONE:
                # best choice wasn't used yet, so link is created
                ref_path_inf.prev = h
            else:
                # best choice was already used by mega[2][mega[1][h].next].prev
                # check which is the better choice
                if fb.buf[1].path_info[ref_path_inf.prev].finaldecis > curr_path_inf.finaldecis:
                    # remove link with prev
                    fb.buf[1].path_info[ref_path_inf.prev].next = NEXT_NONE
                    ref_path_inf.prev = h
                else:
                    curr_path_inf.next = NEXT_NONE

        if curr_path_inf.next != NEXT_NONE:
            count1 += 1

    # end of creation of links with decision check
    print(f"step: {step}, curr: {fb.buf[1].num_parts}, next: {fb.buf[2].num_parts}, links: {count1}, lost: {fb.buf[1].num_parts - count1}, add: {num_added}")

    # for the average of particles and links
    run_info.npart = run_info.npart + fb.buf[1].num_parts
    run_info.nlinks = run_info.nlinks + count1

    fb_next(fb)
    fb_write_frame_from_start(fb, step)

    if step < run_info.seq_par.last - 2:
        fb_read_frame_at_end(fb, step + 3, 0)
    # end of sequence loop
        

import numpy as np

def trackcorr_c_finish(run_info, step):
    range = run_info.seq_par.last - run_info.seq_par.first
    npart, nlinks = run_info.npart / range, run_info.nlinks / range
    print(f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart - nlinks:.1f}")

    fb_next(run_info.fb)
    fb_write_frame_from_start(run_info.fb, step)



def trackback_c(run_info):
    MAX_CANDS = ...
    count1 = count2 = num_added = quali = 0
    Ymin = Ymax = npart = nlinks = 0
    philf = np.zeros((4, MAX_CANDS))
    X = [Vec3d() for _ in range(6)]
    n = [Vec2d() for _ in range(4)]
    v2 = [Vec2d() for _ in range(4)]

    fb = run_info.fb
    seq_par = run_info.seq_par
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    cal = run_info.cal

    # Prime the buffer with first frames
    for step in range(seq_par.last, seq_par.last - 4, -1):
        fb_read_frame_at_end(fb, step, 1)
        fb_next(fb)
    fb_prev(fb)

    # sequence loop
    for step in range(seq_par.last - 1, seq_par.first, -1):
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            # We try to find link only if the forward search failed to.
            if curr_path_inf.next < 0 or curr_path_inf.prev != -1:
                continue

            for j in range(6):
                X[j].init()

            curr_path_inf.inlist = 0

            # 3D-position of current particle
            X[1].copy(curr_path_inf.x)

            # use information from previous to locate new search position
            # and to calculate values for search area
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.next]
            X[0].copy(ref_path_inf.x)
            search_volume_center_moving(ref_path_inf.x, curr_path_inf.x, X[2])

            for j in range(fb.num_cams):
                point_to_pixel(n[j], X[2], cal[j], cpar)

            # calculate searchquader and reprojection in image space
            w = sorted_candidates_in_volume(X[2], n, fb.buf[2], run_info)

            if w is not None:
                count2 += 1

                i = 0
                while w[i].ftnr != TR_UNUSED:
                    ref_path_inf = fb.buf[2].path_info[w[i].ftnr]
                    X[3].copy(ref_path_inf.x)

                    vec_subt(X[1], X[3], diff_pos)
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle_acc(X[1], X[2], X[3], angle, acc)

                        # *********************check link *****************************
                        if acc < tpar.dacc and angle < tpar.dangle or acc < tpar.dacc / 10:
                            dl = (vec_diff_norm(X[1], X[3]) +
                                  vec_diff_norm(X[0], X[1])) / 2
                            quali = w[i].freq
                            rr = (dl / run_info.lmax + acc / tpar.dacc +
                                  angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[i].ftnr)

                    i += 1

            free(w)
            
            # if old wasn't found try to create new particle position from rest
            if tpar.add:
                if curr_path_inf.inlist == 0:
                    quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info)
                    if quali >= 2:
                        # vec_copy(X[3], X[2])
                        in_volume = 0

                        point_position(v2, fb.num_cams, cpar.mm, cal, X[3])

                        # volume check
                        if vpar.X_lay[0] < X[3][0] < vpar.X_lay[1] and \
                        Ymin < X[3][1] < Ymax and \
                        vpar.Zmin_lay[0] < X[3][2] < vpar.Zmax_lay[1]:
                            in_volume = 1

                        vec_subt(X[1], X[3], diff_pos)
                        if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                            angle_acc(X[1], X[2], X[3], angle, acc)

                            if acc < tpar.dacc and angle < tpar.dangle or \
                            acc < tpar.dacc/10:
                                dl = (vec_diff_norm(X[1], X[3]) +
                                    vec_diff_norm(X[0], X[1]))/2
                                rr = (dl/run_info.lmax+acc/tpar.dacc + angle/tpar.dangle)/(quali)
                                register_link_candidate(curr_path_inf, rr, fb.buf[2].num_parts)

                                add_particle(fb.buf[2], X[3], philf)
                        
                        in_volume = 0
        
        # end of h-loop                        
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if curr_path_inf.inlist > 0:
                curr_path_inf.linkdecis = sorted(curr_path_inf.decis[:curr_path_inf.inlist])


    
    
        # create links with decision check 
        count1 = 0
        num_added = 0

        for h in range(fb['buf'][1]['num_parts']):
            curr_path_inf = fb['buf'][1]['path_info'][h]

            if curr_path_inf['inlist'] > 0:
                ref_path_inf = fb['buf'][2]['path_info'][curr_path_inf['linkdecis'][0]]

                if ref_path_inf['prev'] == PREV_NONE and ref_path_inf['next'] == NEXT_NONE:
                    curr_path_inf['finaldecis'] = curr_path_inf['decis'][0]
                    curr_path_inf['prev'] = curr_path_inf['linkdecis'][0]
                    fb['buf'][2]['path_info'][curr_path_inf['prev']]['next'] = h
                    num_added += 1

                if ref_path_inf['prev'] != PREV_NONE and ref_path_inf['next'] == NEXT_NONE:
                    vec_copy(X[0], fb['buf'][0]['path_info'][curr_path_inf['next']]['x'])
                    vec_copy(X[1], curr_path_inf['x'])
                    vec_copy(X[3], ref_path_inf['x'])
                    vec_copy(X[4], fb['buf'][3]['path_info'][ref_path_inf['prev']]['x'])

                    for j in range(3):
                        X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])

                    angle_acc(X[3], X[4], X[5], angle, acc)

                    if (acc < tpar['dacc'] and angle < tpar['dangle']) or (acc < tpar['dacc'] / 10):
                        curr_path_inf['finaldecis'] = curr_path_inf['decis'][0]
                        curr_path_inf['prev'] = curr_path_inf['linkdecis'][0]
                        fb['buf'][2]['path_info'][curr_path_inf['prev']]['next'] = h
                        num_added += 1

            if curr_path_inf['prev'] != PREV_NONE:
                count1 += 1

        npart += fb['buf'][1]['num_parts']
        nlinks += count1

        fb_next(fb)
        fb_write_frame_from_start(fb, step)

        if step > seq_par['first'] + 2:
            fb_read_frame_at_end(fb, step - 3, 1)

        print("step: {}, curr: {}, next: {}, links: {}, lost: {}, add: {}".format(step, fb['buf'][1]['num_parts'],
                                                                                    fb['buf'][2]['num_parts'], count1,
                                                                                    fb['buf'][1]['num_parts'] - count1,
                                                                                    num_added))

    npart /= (seq_par['last'] - seq_par['first'] - 1)
    nlinks /= (seq_par['last'] - seq_par['first'] - 1)

    print("Average over sequence, particles: {:.1f}, links: {:.1f}, lost: {:.1f}".format(npart, nlinks, npart - nlinks))

    fb_next(fb)
    fb_write_frame_from_start(fb, step)

    return nlinks



