import numpy as np
from typing import List, Optional, Sequence, Any
from .vec_utils import (
    Vec2D, Vec3D, vec_subt, vec_dot, vec_norm, vec_diff_norm, vec_scalar_mul, vec_set, vec_copy
)
from .tracking_frame_buf import (
    PathInfo, Corres, PREV_NONE, NEXT_NONE, PRIO_DEFAULT, Target,
    fb_next, fb_prev, fb_write_frame_from_start, fb_read_frame_at_end, Frame, FrameBuffer
)
from .trafo import pixel_to_metric, metric_to_pixel, dist_to_flat
from .imgcoord import img_coord
from .orientation import point_position
from .tracking_run import TrackingRun
from .parameters import ControlPar, TrackPar
from .calibration import Calibration
from .constants import (
    TR_UNUSED, PT_UNUSED, COORD_UNUSED, ADD_PART, CORRES_NONE, MAX_CANDS, MAX_TARGETS
)


class FoundPix:
    """Represents a found pixel candidate for tracking correspondence.

    Attributes:
        ftnr: Feature/track number.
        freq: Frequency of appearance across cameras.
        whichcam: List indicating which cameras detected this candidate.
    """
    ftnr: int
    freq: int
    whichcam: List[int]

    def __init__(
        self, ftnr: int = TR_UNUSED, freq: int = 0, whichcam: Optional[List[int]] = None
    ) -> None:
        self.ftnr = ftnr
        self.freq = freq
        self.whichcam = [0, 0, 0, 0] if whichcam is None else whichcam


def register_link_candidate(path_info: PathInfo, fitness: float, cand: int) -> None:
    """Register a candidate link in the path info structure."""
    path_info.decis[path_info.inlist] = fitness
    path_info.linkdecis[path_info.inlist] = cand
    path_info.inlist += 1


def make_v2(num_cams: int) -> List[Vec2D]:
    """Create a list of Vec2D initialized to zero for each camera."""
    return [Vec2D(0, 0) for _ in range(num_cams)]


def make_philf(num_cams: int) -> List[List[int]]:
    """Create a 2D list for candidate indices, initialized to PT_UNUSED."""
    return [[PT_UNUSED for _ in range(MAX_CANDS)] for _ in range(num_cams)]


def reset_foundpix_array(arr: List[FoundPix], arr_len: int, num_cams: int) -> None:
    """Reset an array of FoundPix objects to default values."""
    for i in range(arr_len):
        arr[i].ftnr = TR_UNUSED
        arr[i].freq = 0
        for cam in range(num_cams):
            arr[i].whichcam[cam] = 0


def copy_foundpix_array(
    dest: List[FoundPix], src: List[FoundPix], arr_len: int, num_cams: int
) -> None:
    """Copy an array of FoundPix objects from src to dest."""
    for i in range(arr_len):
        dest[i].ftnr = src[i].ftnr
        dest[i].freq = src[i].freq
        for cam in range(num_cams):
            dest[i].whichcam[cam] = src[i].whichcam[cam]


def register_closest_neighbs(
    targets: List[Target],
    num_targets: int,
    cam: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    reg: List[FoundPix],
    cpar: ControlPar,
) -> None:
    """Register the closest neighbor candidates for a given camera and search region."""
    all_cands = np.zeros(MAX_CANDS, dtype=np.int32)
    cand = candsearch_in_pix(
        targets, num_targets, cent_x, cent_y, dl, dr, du, dd, all_cands, cpar
    )
    for cand in range(MAX_CANDS):
        if all_cands[cand] == -999:
            reg[cand].ftnr = TR_UNUSED
        else:
            reg[cand].whichcam[cam] = 1
            reg[cand].ftnr = targets[all_cands[cand]].tnr


def search_volume_center_moving(prev_pos: Vec3D, curr_pos: Vec3D) -> Vec3D:
    """Predict the next position in 3D by linear extrapolation."""
    output = vec_scalar_mul(curr_pos, 2)
    return vec_subt(output, prev_pos)


def predict(prev_pos: Vec2D, curr_pos: Vec2D) -> Vec2D:
    """Predict the next 2D position by linear extrapolation."""
    return Vec2D(2 * curr_pos.x - prev_pos.x, 2 * curr_pos.y - prev_pos.y)


def pos3d_in_bounds(pos: Vec3D, bounds: TrackPar) -> bool:
    """Check if a 3D position is within the specified tracking bounds."""
    return (
        bounds.dvxmin < pos.x < bounds.dvxmax
        and bounds.dvymin < pos.y < bounds.dvymax
        and bounds.dvzmin < pos.z < bounds.dvzmax
    )


def angle_acc(start: Vec3D, pred: Vec3D, cand: Vec3D) -> tuple:
    """Compute the angle and acceleration between predicted and candidate positions."""
    v0 = vec_subt(pred, start)
    v1 = vec_subt(cand, start)
    acc = vec_diff_norm(v0, v1)
    if v0.x == -v1.x and v0.y == -v1.y and v0.z == -v1.z:
        angle = 200.0
    elif v0.x == v1.x and v0.y == v1.y and v0.z == v1.z:
        angle = 0.0
    else:
        angle = (200.0 / np.pi) * np.arccos(
            vec_dot(v0, v1) / vec_norm(v0) / vec_norm(v1)
        )
    return angle, acc


def candsearch_in_pix(
    next: List[Target],
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    p: Any,
    cpar: ControlPar,
) -> int:
    """Search for up to four nearest candidate targets in a region."""
    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next[j0].y < cent_y - du:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2
    j0 -= 12
    if j0 < 0:
        j0 = 0
    counter = 0
    p1 = p2 = p3 = p4 = PT_UNUSED
    d1 = d2 = d3 = d4 = 1e20
    for j in range(j0, num_targets):
        if next[j].tnr != TR_UNUSED:
            if next[j].y > cent_y + dd:
                break
            if cent_x - dl < next[j].x < cent_x + dr and cent_y - du < next[j].y < cent_y + dd:
                d = np.sqrt((cent_x - next[j].x) ** 2 + (cent_y - next[j].y) ** 2)
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
    p[0], p[1], p[2], p[3] = p1, p2, p3, p4
    for j in range(4):
        if p[j] != PT_UNUSED:
            counter += 1
    return counter


def candsearch_in_pix_rest(
    next: List[Target],
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    p: List[int],
    cpar: ControlPar,
) -> int:
    """Search for the nearest unused candidate target in a region.

    Args:
        next: List of Target objects to search.
        num_targets: Number of targets in the list.
        cent_x: Center x coordinate of the search region.
        cent_y: Center y coordinate of the search region.
        dl, dr, du, dd: Search region bounds (left, right, up, down).
        p: Output array (list or numpy array) to store found candidate indices.
        cpar: ControlPar object with camera parameters.

    Returns:
        The number of found candidates (0 or 1).
    """
    # p is an output array (list or numpy array) where the index of the found candidate is stored.
    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next[j0].y < cent_y - du:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2
    j0 -= 12
    if j0 < 0:
        j0 = 0
    counter = 0
    dmin = 1e20
    p[0] = PT_UNUSED
    for j in range(j0, num_targets):
        if next[j].tnr == TR_UNUSED:
            if next[j].y > cent_y + dd:
                break
            if cent_x - dl < next[j].x < cent_x + dr and cent_y - du < next[j].y < cent_y + dd:
                d = np.sqrt((cent_x - next[j].x) ** 2 + (cent_y - next[j].y) ** 2)
                if d < dmin:
                    dmin = d
                    p[0] = j
    if p[0] != PT_UNUSED:
        counter += 1
    return counter


def searchquader(
    point: Vec3D,
    xr: np.ndarray,
    xl: np.ndarray,
    yd: np.ndarray,
    yu: np.ndarray,
    tpar: TrackPar,
    cpar: ControlPar,
    cal: List[Calibration],
) -> None:
    """Project a 3D cuboid (search volume) to image space and compute search bounds for each camera."""
    mins = vec_set(tpar.dvxmin, tpar.dvymin, tpar.dvzmin)
    maxes = vec_set(tpar.dvxmax, tpar.dvymax, tpar.dvzmax)
    quader = [vec_copy(point) for _ in range(8)]
    for pt in range(8):
        for dim in range(3):
            if pt & (1 << dim):
                quader[pt][dim] += maxes[dim]
            else:
                quader[pt][dim] += mins[dim]
    for i in range(cpar.num_cams):
        xr[i] = 0
        xl[i] = cpar.imx
        yd[i] = 0
        yu[i] = cpar.imy
        center = point_to_pixel(point, cal[i], cpar)
        for pt in range(8):
            corner = point_to_pixel(quader[pt], cal[i], cpar)
            if corner.x < xl[i]:
                xl[i] = corner.x
            if corner.y < yu[i]:
                yu[i] = corner.y
            if corner.x > xr[i]:
                xr[i] = corner.x
            if corner.y > yd[i]:
                yd[i] = corner.y
        if xl[i] < 0:
            xl[i] = 0
        if yu[i] < 0:
            yu[i] = 0
        if xr[i] > cpar.imx:
            xr[i] = cpar.imx
        if yd[i] > cpar.imy:
            yd[i] = cpar.imy
        xr[i] -= center.x
        xl[i] = center.x - xl[i]
        yd[i] -= center.y
        yu[i] = center.y - yu[i]


def sort_candidates_by_freq(item: List[FoundPix], num_cams: int) -> int:
    """Sort candidate found pixels by frequency and return the number of unique candidates."""
    different = 0
    for i in range(num_cams * MAX_CANDS):
        for j in range(num_cams):
            for m in range(MAX_CANDS):
                if item[i].ftnr == item[4 * j + m].ftnr:
                    item[i].whichcam[j] = 1
    for i in range(num_cams * MAX_CANDS):
        for j in range(num_cams):
            if item[i].whichcam[j] == 1 and item[i].ftnr != TR_UNUSED:
                item[i].freq += 1
    for i in range(1, num_cams * MAX_CANDS):
        for j in range(num_cams * MAX_CANDS - 1, i - 1, -1):
            if item[j - 1].freq < item[j].freq:
                item[j - 1], item[j] = item[j], item[j - 1]
    for i in range(num_cams * MAX_CANDS):
        for j in range(i + 1, num_cams * MAX_CANDS):
            if item[i].ftnr == item[j].ftnr or item[j].freq < 2:
                item[j].freq = 0
                item[j].ftnr = TR_UNUSED
    for i in range(1, num_cams * MAX_CANDS):
        for j in range(num_cams * MAX_CANDS - 1, i - 1, -1):
            if item[j - 1].freq < item[j].freq:
                item[j - 1], item[j] = item[j], item[j - 1]
    for i in range(num_cams * MAX_CANDS):
        if item[i].freq != 0:
            different += 1
    return different


def sort(arr: List[Any]) -> List[Any]:
    """Sort an array in place using bubble sort."""
    flag = 0
    while True:
        flag = 0
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                flag = 1
        if flag == 0:
            break


def point_to_pixel(point: Vec3D, cal: Calibration, cpar: ControlPar) -> Vec2D:
    """Project a 3D point to 2D pixel coordinates for a given camera."""
    x, y = img_coord(point, cal, cpar.mm)
    return metric_to_pixel(x, y, cpar)


def sorted_candidates_in_volume(
    center: Vec3D,
    center_proj: List[Vec2D],
    frm: Frame,
    run: TrackingRun,
) -> Optional[List[FoundPix]]:
    """Find and sort candidate targets in a 3D search volume across all cameras."""
    num_cams = frm.num_cams
    points: List[FoundPix] = [FoundPix(TR_UNUSED, 0, [0] * num_cams) for _ in range(num_cams * MAX_CANDS)]
    reset_foundpix_array(points, num_cams * MAX_CANDS, num_cams)
    right = np.zeros(num_cams)
    left = np.zeros(num_cams)
    down = np.zeros(num_cams)
    up = np.zeros(num_cams)
    searchquader(center, right, left, down, up, run.tpar, run.cpar, run.cal)
    for cam in range(num_cams):
        register_closest_neighbs(
            frm.targets[cam],
            frm.num_targets[cam],
            cam,
            center_proj[cam].x,
            center_proj[cam].y,
            left[cam],
            right[cam],
            up[cam],
            down[cam],
            points[cam * MAX_CANDS :],
            run.cpar,
        )
    num_cands = sort_candidates_by_freq(points, num_cams)
    if num_cands > 0:
        return points[:num_cands]
    else:
        return None


def assess_new_position(
    pos: Vec3D,
    targ_pos: List[Vec2D],
    cand_inds: np.ndarray,
    frm: Frame,
    run: TrackingRun,
) -> int:
    """Assess the new 3D position by searching for corresponding 2D targets in all cameras.

    Args:
        pos: The 3D position to assess.
        targ_pos: List to store the corresponding 2D positions for each camera.
        cand_inds: List of candidate indices for each camera.
        frm: The current frame buffer containing target information (expected type: TrackingFrameBuf).
        run: The tracking run configuration.

    Returns:
        The number of valid cameras where a target was found.
    """
    left = right = up = down = ADD_PART
    for cam in range(run.cpar.num_cams):
        targ_pos[cam] = Vec2D(COORD_UNUSED, COORD_UNUSED)
    for cam in range(run.cpar.num_cams):
        pixel = point_to_pixel(pos, run.cal[cam], run.cpar)
        num_cands = candsearch_in_pix_rest(
            frm.targets[cam],
            frm.num_targets[cam],
            pixel.x,
            pixel.y,
            left,
            right,
            up,
            down,
            cand_inds[cam],
            run.cpar,
        )
        if num_cands > 0:
            _ix = cand_inds[cam][0]
            targ_pos[cam] = Vec2D(frm.targets[cam][_ix].x, frm.targets[cam][_ix].y)
    valid_cams = 0
    for cam in range(run.cpar.num_cams):
        if targ_pos[cam].x != COORD_UNUSED and targ_pos[cam].y != COORD_UNUSED:
            targ_pos[cam] = pixel_to_metric(
                targ_pos[cam].x, targ_pos[cam].y, run.cpar
            )
            targ_pos[cam] = dist_to_flat(
                targ_pos[cam].x, targ_pos[cam].y, run.cal[cam], run.flatten_tol
            )
            valid_cams += 1
    return valid_cams


def add_particle(frm: Frame, pos: Vec3D, cand_inds: np.ndarray) -> None:
    """Add a new particle to the frame at the specified position."""
    num_parts = frm.num_parts
    ref_path_inf = PathInfo(pos, PREV_NONE, NEXT_NONE, 0, np.zeros(MAX_CANDS), np.zeros(MAX_CANDS), 0)
    frm.path_info.append(ref_path_inf)
    ref_corres = Corres(num_parts, [CORRES_NONE] * frm.num_cams)
    frm.correspond.append(ref_corres)
    for cam in range(frm.num_cams):
        if cand_inds[cam][0] != PT_UNUSED:
            _ix = cand_inds[cam][0]
            frm.targets[cam][_ix].tnr = num_parts
            ref_corres.p[cam] = _ix
    frm.num_parts += 1


def trackcorr_c_loop(run_info: TrackingRun, step: int) -> None:
    """Main tracking loop for updating particle correspondences between frames."""
    fb = run_info.fb
    cal = run_info.cal
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    curr_targets = fb.buf[1].targets
    orig_parts = fb.buf[1].num_parts
    v2 = make_v2(fb.num_cams)
    philf = make_philf(fb.num_cams)
    for h in range(orig_parts):
        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres = fb.buf[1].correspond[h]
        curr_path_inf.inlist = 0
        X = [Vec3D(0, 0, 0) for _ in range(6)]
        X[1] = curr_path_inf.x
        if curr_path_inf.prev >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev]
            X[0] = ref_path_inf.x
            X[2] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)
            v1 = [point_to_pixel(X[2], cal[j], cpar) for j in range(fb.num_cams)]
        else:
            X[2] = X[1]
            v1 = [point_to_pixel(X[2], cal[j], cpar) if curr_corres.p[j] == CORRES_NONE else Vec2D(curr_targets[j][curr_corres.p[j]].x, curr_targets[j][curr_corres.p[j]].y) for j in range(fb.num_cams)]
        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2], run_info)
        if w is None:
            continue
        mm = 0
        while w[mm].ftnr != TR_UNUSED:
            ref_path_inf = fb.buf[2].path_info[w[mm].ftnr]
            X[3] = ref_path_inf.x
            if curr_path_inf.prev >= 0:
                X[5] = Vec3D(0.5 * (5.0 * X[3].x - 4.0 * X[1].x + X[0].x), 0.5 * (5.0 * X[3].y - 4.0 * X[1].y + X[0].y), 0.5 * (5.0 * X[3].z - 4.0 * X[1].z + X[0].z))
            else:
                X[5] = search_volume_center_moving(X[1], X[3])
            v1 = [point_to_pixel(X[5], cal[j], cpar) for j in range(fb.num_cams)]
            wn = sorted_candidates_in_volume(X[5], v1, fb.buf[3], run_info)
            if wn is not None:
                kk = 0
                while wn[kk].ftnr != TR_UNUSED:
                    ref_path_inf = fb.buf[3].path_info[wn[kk].ftnr]
                    X[4] = ref_path_inf.x
                    diff_pos = vec_subt(X[4], X[3])
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle1, acc1 = angle_acc(X[3], X[4], X[5])
                        if curr_path_inf.prev >= 0:
                            angle0, acc0 = angle_acc(X[1], X[2], X[3])
                        else:
                            acc0, angle0 = acc1, angle1
                        acc = (acc0 + acc1) / 2
                        angle = (angle0 + angle1) / 2
                        quali = wn[kk].freq + w[mm].freq
                        if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                            dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[mm].ftnr)
                    kk += 1
            quali = assess_new_position(X[5], v2, philf, fb.buf[3], run_info)
            if quali >= 2:
                in_volume = 0
                dl = point_position(v2, cpar.num_cams, cpar.mm, cal, X[4])
                if vpar.X_lay[0] < X[4].x < vpar.X_lay[1] and run_info.ymin < X[4].y < run_info.ymax and vpar.Zmin_lay[0] < X[4].z < vpar.Zmax_lay[1]:
                    in_volume = 1
                diff_pos = vec_subt(X[3], X[4])
                if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                        dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])) / 2
                        rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / (quali + w[mm].freq)
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr)
                        if tpar.add:
                            add_particle(fb.buf[3], X[4], philf)
                            num_added += 1
            if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
                diff_pos = vec_subt(X[3], X[1])
                if pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[1], X[2], X[3])
                    if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                        quali = w[mm].freq
                        dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                        rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr)
            mm += 1
        if tpar.add and curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
            quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info)
            if quali >= 2:
                X[3] = X[2]
                in_volume = 0
                dl = point_position(v2, fb.num_cams, cpar.mm, cal, X[3])
                if vpar.X_lay[0] < X[3].x < vpar.X_lay[1] and run_info.ymin < X[3].y < run_info.ymax and vpar.Zmin_lay[0] < X[3].z < vpar.Zmax_lay[1]:
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
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.inlist > 0:
                sort(curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis)
                curr_path_inf.finaldecis = curr_path_inf.decis[0]
                curr_path_inf.next = curr_path_inf.linkdecis[0]
        count1 = 0
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.inlist > 0:
                ref_path_inf = fb.buf[2].path_info[curr_path_inf.next]
                if ref_path_inf.prev == PREV_NONE:
                    ref_path_inf.prev = h
                else:
                    if fb.buf[1].path_info[ref_path_inf.prev].finaldecis > curr_path_inf.finaldecis:
                        fb.buf[1].path_info[ref_path_inf.prev].next = NEXT_NONE
                        ref_path_inf.prev = h
                    else:
                        curr_path_inf.next = NEXT_NONE
            if curr_path_inf.next != NEXT_NONE:
                count1 += 1
        print(f"step: {step}, curr: {fb.buf[1].num_parts}, next: {fb.buf[2].num_parts}, links: {count1}, lost: {fb.buf[1].num_parts - count1}, add: {num_added}")
        run_info.npart += fb.buf[1].num_parts
        run_info.nlinks += count1
        fb_next(fb)
        fb_write_frame_from_start(fb, step)
        if step < run_info.seq_par.last - 2:
            fb_read_frame_at_end(fb, step + 3, 0)

def trackcorr_c_finish(run_info: TrackingRun, step: int) -> None:
    range_ = run_info.seq_par.last - run_info.seq_par.first
    npart = run_info.npart / range_
    nlinks = run_info.nlinks / range_
    print(f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart - nlinks:.1f}")
    fb_next(run_info.fb)
    fb_write_frame_from_start(run_info.fb, step)

def trackback_c(run_info: TrackingRun) -> int:
    seq_par = run_info.seq_par
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    fb = run_info.fb
    cal = run_info.cal
    v2 = make_v2(fb.num_cams)
    philf = make_philf(fb.num_cams)
    for step in range(seq_par.last, seq_par.last - 4, -1):
        fb_read_frame_at_end(fb, step, 1)
        fb_next(fb)
    fb_prev(fb)
    npart = nlinks = 0
    for step in range(seq_par.last - 1, seq_par.first, -1):
        v2 = make_v2(fb.num_cams)
        philf = make_philf(fb.num_cams)
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.next < 0 or curr_path_inf.prev != -1:
                continue
            X = [Vec3D(0, 0, 0) for _ in range(6)]
            curr_path_inf.inlist = 0
            X[1] = curr_path_inf.x
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.next]
            X[0] = ref_path_inf.x
            X[2] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)
            n = [point_to_pixel(X[2], cal[j], cpar) for j in range(fb.num_cams)]
            w = sorted_candidates_in_volume(X[2], n, fb.buf[2], run_info)
            if w is not None:
                i = 0
                while w[i].ftnr != TR_UNUSED:
                    ref_path_inf = fb.buf[2].path_info[w[i].ftnr]
                    X[3] = ref_path_inf.x
                    diff_pos = vec_subt(X[1], X[3])
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                            dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                            quali = w[i].freq
                            rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[i].ftnr)
                    i += 1
            if tpar.add and curr_path_inf.inlist == 0:
                quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info)
                if quali >= 2:
                    in_volume = 0
                    point_position(v2, fb.num_cams, cpar.mm, cal, X[3])
                    if vpar.X_lay[0] < X[3].x < vpar.X_lay[1] and run_info.ymin < X[3].y < run_info.ymax and vpar.Zmin_lay[0] < X[3].z < vpar.Zmax_lay[1]:
                        in_volume = 1
                    diff_pos = vec_subt(X[1], X[3])
                    if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                            dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, fb.buf[2].num_parts)
                            add_particle(fb.buf[2], X[3], philf)
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.inlist > 0:
                sort(curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis)
        count1 = num_added = 0
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.inlist > 0:
                ref_path_inf = fb.buf[2].path_info[curr_path_inf.linkdecis[0]]
                if ref_path_inf.prev == PREV_NONE and ref_path_inf.next == NEXT_NONE:
                    curr_path_inf.finaldecis = curr_path_inf.decis[0]
                    curr_path_inf.prev = curr_path_inf.linkdecis[0]
                    fb.buf[2].path_info[curr_path_inf.prev].next = h
                    num_added += 1
                if ref_path_inf.prev != PREV_NONE and ref_path_inf.next == NEXT_NONE:
                    X[0] = fb.buf[0].path_info[curr_path_inf.next].x
                    X[1] = curr_path_inf.x
                    X[3] = ref_path_inf.x
                    X[4] = fb.buf[3].path_info[ref_path_inf.prev].x
                    X[5] = Vec3D(0.5 * (5.0 * X[3].x - 4.0 * X[1].x + X[0].x), 0.5 * (5.0 * X[3].y - 4.0 * X[1].y + X[0].y), 0.5 * (5.0 * X[3].z - 4.0 * X[1].z + X[0].z))
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                        curr_path_inf.finaldecis = curr_path_inf.decis[0]
                        curr_path_inf.prev = curr_path_inf.linkdecis[0]
        count1 = num_added = 0
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]
            if curr_path_inf.inlist > 0:
                ref_path_inf = fb.buf[2].path_info[curr_path_inf.linkdecis[0]]
                if ref_path_inf.prev == PREV_NONE and ref_path_inf.next == NEXT_NONE:
                    curr_path_inf.finaldecis = curr_path_inf.decis[0]
                    curr_path_inf.prev = curr_path_inf.linkdecis[0]
                    fb.buf[2].path_info[curr_path_inf.prev].next = h
                    num_added += 1
                if ref_path_inf.prev != PREV_NONE and ref_path_inf.next == NEXT_NONE:
                    X[0] = fb.buf[0].path_info[curr_path_inf.next].x
                    X[1] = curr_path_inf.x
                    X[3] = ref_path_inf.x
                    X[4] = fb.buf[3].path_info[ref_path_inf.prev].x
                    X[5] = Vec3D(0.5 * (5.0 * X[3].x - 4.0 * X[1].x + X[0].x), 0.5 * (5.0 * X[3].y - 4.0 * X[1].y + X[0].y), 0.5 * (5.0 * X[3].z - 4.0 * X[1].z + X[0].z))
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
                        curr_path_inf.finaldecis = curr_path_inf.decis[0]
                        curr_path_inf.prev = curr_path_inf.linkdecis[0]
        print(f"step: {step}, curr: {fb.buf[1].num_parts}, next: {fb.buf[2].num_parts}, links: {count1}, lost: {fb.buf[1].num_parts - count1}, add: {num_added}")
        npart += fb.buf[1].num_parts
        nlinks += count1
        fb_next(fb)
        fb_write_frame_from_start(fb, step)
        if step > seq_par.first + 2:
            fb_read_frame_at_end(fb, step - 3, 1)
    npart /= (seq_par.last - seq_par.first - 1)
    nlinks /= (seq_par.last - seq_par.first - 1)
    print(f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart - nlinks:.1f}")
    fb_next(fb)
    fb_write_frame_from_start(fb, step)
    return nlinks
