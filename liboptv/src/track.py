def add_particle(frm, pos, cand_inds):
    num_parts = frm.num_parts # Get the number of existing particles
    ref_path_inf = frm.path_info[num_parts] # Get the new particle's path_info
    ref_path_inf.x = pos # Copy the new particle's position into path_info
    reset_links(ref_path_inf) # Reset path_info's links array

    ref_corres = frm.correspond[num_parts] # Get the new particle's correspond struct
    ref_targets = frm.targets # Get the targets array
    for cam in range(frm.num_cams): # Loop over all cameras
        ref_corres.p[cam] = CORRES_NONE # Set the correspond struct's p[cam] to CORRES_NONE
        
        # We always take the 1st candidate, apparently. Why did we fetch 4?
        if cand_inds[cam][0] != PT_UNUSED: # Check if the first candidate for this camera is valid
            _ix = cand_inds[cam][0] # Get the index of the first candidate for this camera
            ref_targets[cam][_ix].tnr = num_parts # Set the target's tnr to the new particle's index
            ref_corres.p[cam] = _ix # Set the correspond struct's p[cam] to the first candidate index
            ref_corres.nr = num_parts # Set the correspond struct's nr to the new particle's index
    
    frm.num_parts += 1 # Increment the number of particles in the frame
    
    
def sorted_candidates_in_volume(center, center_proj, frm, run):
    """
    sorted_candidates_in_volume() receives a volume center and produces a list of candidates for the next particle in
    that volume, sorted by the candidates' number of appearances as 2D targets.

    Arguments:
    center: vec3d - the 3D midpoint-position of the search volume
    center_proj: vec2d[] - projections of the center on the cameras, pixel coordinates.
    frm: frame - the frame holding targets for the search.
    run: tracking_run - the parameter collection we need for determining search region. The same object used
        throughout the tracking code.

    Returns:
    foundpix[] - a newly-allocated buffer of foundpix items, denoting for each item its particle number and quality
        parameters. The buffer is terminated by one extra item with ftnr set to TR_UNUSED
    """
    num_cams = frm.num_cams
    points = [None] * num_cams * MAX_CANDS
    reset_foundpix_array(points, num_cams * MAX_CANDS, num_cams)

    # Search limits in image space
    right, left, down, up = searchquader(center, run.tpar, run.cpar, run.cal)

    # search in pix for candidates in the next time step
    for cam in range(num_cams):
        register_closest_neighbs(frm.targets[cam], frm.num_targets[cam], cam, center_proj[cam][0],
                                 center_proj[cam][1], left[cam], right[cam], up[cam], down[cam],
                                 points[cam * MAX_CANDS: (cam + 1) * MAX_CANDS], run.cpar)

    # fill and sort candidate struct
    num_cands = sort_candidates_by_freq(points, num_cams)
    if num_cands > 0:
        points = points[:num_cands + 1]
        points[num_cands].ftnr = TR_UNUSED
        return points
    else:
        return None


TR_UNUSED = -1

def track_forward_start(tr):
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


import math

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
from typing import List

def searchquader(point: np.ndarray, tpar: np.ndarray, cpar: np.ndarray, cal: List[np.ndarray]) -> List[np.ndarray]:
    # Define some useful variables
    num_cams = cpar.shape[0]
    mins = np.array([tpar[:,0], tpar[:,1], tpar[:,2]]).T
    maxes = np.array([tpar[:,3], tpar[:,4], tpar[:,5]]).T
    quader = np.zeros((8, 3, num_cams))
    xr, xl, yd, yu = np.zeros(num_cams), np.full(num_cams, cpar[:,3]), np.zeros(num_cams), np.full(num_cams, cpar[:,4])
    
    # Compute the 3D positions of the search volume (eight corners of a box)
    for dim in range(3):
        for pt in range(8):
            quader[pt, dim, :] = point[:, dim]
            mask = (pt & 1 << dim) != 0
            quader[pt, dim, mask] += maxes[:, dim][mask]
            quader[pt, dim, ~mask] += mins[:, dim][~mask]
    
    # Compute the search area in each camera
    for i in range(num_cams):
        # Pixel position of the search center
        center = cal[i].world_to_image(point.T).T
        
        # Mark 8 corners of the search region in pixels
        corners = cal[i].world_to_image(quader[..., i].T).T
        
        # Update the limits of the search area
        xl[i] = np.minimum(xl[i], np.min(corners[:, 0]))
        yu[i] = np.minimum(yu[i], np.min(corners[:, 1]))
        xr[i] = np.maximum(xr[i], np.max(corners[:, 0]))
        yd[i] = np.maximum(yd[i], np.max(corners[:, 1]))
        
        # Ensure that the search area does not exceed the image boundaries
        xl[i] = np.clip(xl[i], 0, cpar[i, 3])
        yu[i] = np.clip(yu[i], 0, cpar[i, 4])
        xr[i] = np.clip(xr[i], 0, cpar[i, 3])
        yd[i] = np.clip(yd[i], 0, cpar[i, 4])
        
        # Compute the distances from the search center to the limits
        xr[i] = xr[i] - center[:, 0]
        xl[i] = center[:, 0] - xl[i]
        yd[i] = yd[i] - center[:, 1]
        yu[i] = center[:, 1] - yu[i]
        
    # Return the arrays xr, xl, yd, yu for the search of a quader (cuboid),
    # given in pixel distances, relative to the point of search
    return [xr, xl, yd, yu]


import numpy as np

def sort_candidates_by_freq(item, num_cams):
    # Reshape the item array into a 3D array
    item_3d = item.reshape((num_cams, -1, MAX_CANDS))

    # where what was found
    whichcam = np.any(np.equal(item_3d[:, :, :, None], item_3d[:, None, :, :]), axis=2)
    item_3d[:, :, :, None] = whichcam

    # how often was ftnr found
    freq = np.sum(np.logical_and(item_3d[:, :, :, None], item_3d[:, None, :, :, None])[:, :, :, :, 0], axis=(1, 2))
    item_3d[:, :, :, 0] = freq

    # sort freq
    sort_idx = np.argsort(-freq.ravel())
    item_3d = item_3d[:, sort_idx // MAX_CANDS, sort_idx % MAX_CANDS]

    # prune the duplicates or those that are found only once
    keep_idx = np.where(np.logical_and(item_3d[:, :, :, 0] != 0, np.greater_equal(item_3d[:, :, :, 0], 2)))
    item_3d[:, :, :, 0] = 0
    item_3d[keep_idx] = item_3d[keep_idx][:, :, [0]]

    # sort freq again on the clean dataset
    freq = np.sum(item_3d[:, :, :, 0], axis=(1, 2))
    sort_idx = np.argsort(-freq)
    item_3d = item_3d[:, sort_idx // MAX_CANDS, sort_idx % MAX_CANDS]

    different = np.count_nonzero(freq)
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
    x, y = img_coord(point, cal, cpar['mm'])
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




import numpy as np

def trackcorr_c_loop(run_info, step):
    MAX_CANDS = 1000
    # sequence loop
    in_volume = 0
    philf = np.zeros((4, MAX_CANDS), dtype=np.int32)
    count1, count2, count3, num_added = 0, 0, 0, 0
    quali = 0
    diff_pos, X = np.zeros((3,), dtype=np.float64), np.zeros((6, 3), dtype=np.float64)     
    # 7 reference points used in the algorithm, TODO: check if can reuse some
    angle, acc, angle0, acc0, dl = 0, 0, 0, 0, 0
    angle1, acc1 = 0, 0
    v1, v2 = np.zeros((4, 2), dtype=np.float64), np.zeros((4, 2), dtype=np.float64) 
    # volume center projection on cameras 
    rr = 0

    # Shortcuts to inside current frame 
    curr_path_inf, ref_path_inf = None, None
    curr_corres = None
    curr_targets = None
    _ix = 0 # For use in any of the complex index expressions below 
    orig_parts = 0 # avoid infinite loop with particle addition set 

    # Shortcuts into the tracking_run struct 
    cal, fb, tpar, vpar, cpar = None, None, None, None, None

    w, wn = None, None 
    count1 = 0
    num_added = 0

    fb = run_info['fb']
    cal = run_info['cal']
    tpar = run_info['tpar']
    vpar = run_info['vpar']
    cpar = run_info['cpar']
    curr_targets = fb.buf[1].targets

    # try to track correspondences from previous 0 - corp, variable h 
    orig_parts = fb.buf[1].num_parts
    for h in range(orig_parts):
        X.fill(0)
        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres = fb.buf[1].correspond[h]
        curr_path_inf.inlist = 0

        # 3D-position 
        X[1] = curr_path_inf.x

        # use information from previous to locate new search position
        # and to calculate values for search area 
        if curr_path_inf.prev >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev]
            X[0] = ref_path_inf.x
            search_volume_center_moving(ref_path_inf.x, curr_path_inf.x, X[2])

            for j in range(fb.num_cams):
                point_to_pixel(v1[j], X[2], cal[j], cpar)
        else:
            X[2] = X[1]
            for j in range(fb.num_cams):
                if curr_corres.p[j] == CORRES_NONE:
                    point_to_pixel(v1[j], X[2], cal[j], cpar)
                else:
                    _ix = curr_corres.p[j]
                    v1[j, 0] = curr_targets[j][_ix].x
                    v1[j, 1] = curr_targets[j][_ix].y

        # calculate search cuboid and reproject it to the image space 
        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2],


