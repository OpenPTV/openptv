import numpy as np
from typing import List, Any
from pyoptv.calibration import Calibration
from pyoptv.parameters import ControlPar, VolumePar
from pyoptv.tracking_frame_buf import Frame, Target
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .epi import Coord2D, epi_mm, find_candidate
from .constants import MAXCAND, PT_UNUSED

nmax = 202400

class NTupel:
    def __init__(self, indices: List[int] = None, corr: float = 0.0):
        if indices is None:
            self.p: List[int] = []
        else:
            self.p: List[int] = indices
        self.corr: float = corr

class Correspond:
    def __init__(self, p1: int = PT_UNUSED, n: int = 0, dist: np.ndarray = None, corr: np.ndarray = None, p2: np.ndarray = None):
        self.p1: int = p1  # Add p1 attribute for compatibility with tests
        self.n: int = n
        self.p2: np.ndarray = np.zeros(MAXCAND, dtype=np.int32) if p2 is None else p2.astype(np.int32)
        self.dist: np.ndarray = np.zeros(MAXCAND, dtype=np.float64) if dist is None else dist.astype(np.float64)
        self.corr: np.ndarray = np.zeros(MAXCAND, dtype=np.float64) if corr is None else corr.astype(np.float64)

    def __repr__(self):
        return f"Correspond(p1={self.p1}, n={self.n}, len(p2)={len(self.p2)})"

def quicksort_con(con: List[Correspond]) -> None:
    if len(con) > 0:
        qs_con(con, 0, len(con) - 1)

def qs_con(con, left, right):
    i = left
    j = right
    xm = con[(left + right) // 2].corr

    while i <= j:
        while con[i].corr > xm and i < right:
            i += 1
        while xm > con[j].corr and j > left:
            j -= 1

        if i <= j:
            con[i], con[j] = con[j], con[i]
            i += 1
            j -= 1

    if left < j:
        qs_con(con, left, j)
    if i < right:
        qs_con(con, i, right)

def quicksort_target_y(pix: List[Target]) -> None:
    qs_target_y(pix, 0, len(pix) - 1)

def qs_target_y(pix, left, right):
    i = left
    j = right
    ym = pix[(left + right) // 2].y

    while i <= j:
        while pix[i].y < ym and i < right:
            i += 1
        while ym < pix[j].y and j > left:
            j -= 1

        if i <= j:
            pix[i], pix[j] = pix[j], pix[i]
            i += 1
            j -= 1

    if left < j:
        qs_target_y(pix, left, j)
    if i < right:
        qs_target_y(pix, i, right)

def quicksort_coord2d_x(crd: List[Coord2D]) -> None:
    qs_coord2d_x(crd, 0, len(crd) - 1)

def qs_coord2d_x(crd, left, right):
    i = left
    j = right
    xm = crd[(left + right) // 2].x

    while i <= j:
        while crd[i].x < xm and i < right:
            i += 1
        while xm < crd[j].x and j > left:
            j -= 1

        if i <= j:
            crd[i], crd[j] = crd[j], crd[i]
            i += 1
            j -= 1

    if left < j:
        qs_coord2d_x(crd, left, j)
    if i < right:
        qs_coord2d_x(crd, i, right)

def safely_allocate_target_usage_marks(num_cams: int) -> np.ndarray:
    try:
        tusage = np.zeros((num_cams, nmax), dtype=np.int32)
        return tusage
    except MemoryError:
        return None

# def deallocate_target_usage_marks(tusage: np.ndarray) -> None:
#     del tusage

def safely_allocate_adjacency_lists(num_cams: int, target_counts: List[int]) -> List[List[List[Correspond]]] | None:
    try:
        lists = [[[] for _ in range(num_cams)] for _ in range(num_cams)]
        for c1 in range(num_cams - 1):
            for c2 in range(c1 + 1, num_cams):
                lists[c1][c2] = [
                    Correspond(n=0, p1=PT_UNUSED)
                    for _ in range(target_counts[c1])
                ]
        return lists
    except MemoryError:
        return None

# def deallocate_adjacency_lists(lists: List[List[List[Correspond]]]) -> None:
#     del lists

def four_camera_matching(
    lists: List[List[List[Any]]],
    num_targets: int,
    corrmin: float,
    scratch: List[NTupel],
    scratch_size: int
) -> int:
    matched = 0
    for i in range(num_targets):
        p1 = lists[0][1][i].p1
        for j in range(lists[0][1][i].n):
            for k in range(lists[0][2][i].n):
                for l in range(lists[0][3][i].n):
                    p2 = lists[0][1][i].p2[j]
                    p3 = lists[0][2][i].p2[k]
                    p4 = lists[0][3][i].p2[l]
                    for m in range(lists[1][2][p2].n):
                        p31 = lists[1][2][p2].p2[m]
                        if p3 != p31:
                            continue
                        for n in range(lists[1][3][p2].n):
                            p41 = lists[1][3][p2].p2[n]
                            if p4 != p41:
                                continue
                            for o in range(lists[2][3][p3].n):
                                p42 = lists[2][3][p3].p2[o]
                                if p4 != p42:
                                    continue

                                corr = (lists[0][1][i].corr[j]
                                        + lists[0][2][i].corr[k]
                                        + lists[0][3][i].corr[l]
                                        + lists[1][2][p2].corr[m]
                                        + lists[1][3][p2].corr[n]
                                        + lists[2][3][p3].corr[o]) / (
                                               lists[0][1][i].dist[j]
                                               + lists[0][2][i].dist[k]
                                               + lists[0][3][i].dist[l]
                                               + lists[1][2][p2].dist[m]
                                               + lists[1][3][p2].dist[n]
                                               + lists[2][3][p3].dist[o])

                                if corr <= corrmin:
                                    continue

                                scratch[matched] = NTupel([p1, p2, p3, p4], corr)
                                matched += 1
                                if matched == scratch_size:
                                    print("Overflow in correspondences.")
                                    return matched
    return matched

def three_camera_matching(
    lists: List[List[List[Any]]],
    num_targets: int,
    num_targets_arr: List[int],
    corrmin: float,
    scratch: List[NTupel],
    scratch_size: int,
    tusage: np.ndarray
) -> int:
    matched = 0
    num_cams = len(lists)
    for i1 in range(num_cams - 2):
        for i in range(num_targets_arr[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = lists[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                for j in range(lists[i1][i2][i].n):
                    p2 = lists[i1][i2][i].p2[j]
                    if p2 > nmax or tusage[i2][p2] > 0:
                        continue

                    for i3 in range(i2 + 1, num_cams):
                        for k in range(lists[i1][i3][i].n):
                            p3 = lists[i1][i3][i].p2[k]
                            if p3 > nmax or tusage[i3][p3] > 0:
                                continue

                            for m in range(lists[i2][i3][p2].n):
                                if p3 != lists[i2][i3][p2].p2[m]:
                                    continue

                                corr = (lists[i1][i2][i].corr[j]
                                        + lists[i1][i3][i].corr[k]
                                        + lists[i2][i3][p2].corr[m]) / (
                                               lists[i1][i2][i].dist[j]
                                               + lists[i1][i3][i].dist[k]
                                               + lists[i2][i3][p2].dist[m])

                                if corr <= corrmin:
                                    continue

                                scratch[matched] = NTupel([-2] * num_cams, corr)
                                scratch[matched].p[i1] = p1
                                scratch[matched].p[i2] = p2
                                scratch[matched].p[i3] = p3
                                matched += 1
                                if matched == scratch_size:
                                    print("Overflow in correspondences.")
                                    return matched
    return matched

def consistent_pair_matching(
    lists: List[List[List[Any]]],
    num_targets: int,
    num_targets_arr: List[int],
    corrmin: float,
    scratch: List[NTupel],
    scratch_size: int,
    tusage: np.ndarray
) -> int:
    matched = 0
    num_cams = len(lists)
    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(num_targets_arr[i1]):
                p1 = lists[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                if lists[i1][i2][i].n != 1:
                    continue

                p2 = lists[i1][i2][i].p2[0]
                if p2 > nmax or tusage[i2][p2] > 0:
                    continue

                corr = lists[i1][i2][i].corr[0] / lists[i1][i2][i].dist[0]
                if corr <= corrmin:
                    continue

                scratch[matched] = NTupel([-2] * num_cams, corr)
                scratch[matched].p[i1] = p1
                scratch[matched].p[i2] = p2
                matched += 1
                if matched == scratch_size:
                    print("Overflow in correspondences.")
                    return matched
    return matched

def match_pairs(
    lists: List[List[List[Correspond]]],
    corrected: List[List[Target]],
    frm: Frame,
    vpar: VolumePar,
    cpar: ControlPar,
    calib: List[Calibration]
) -> None:
    for i1 in range(cpar.num_cams - 1):
        for i2 in range(i1 + 1, cpar.num_cams):
            for i in range(frm.num_targets[i1]):
                if corrected[i1][i].x == -999:
                    continue

                xa12, ya12, xb12, yb12 = epi_mm(corrected[i1][i].x, corrected[i1][i].y,
                                                calib[i1], calib[i2], cpar.mm,
                                                vpar)

                lists[i1][i2][i].p1 = i
                pt1 = corrected[i1][i].pnr

                count, cand = find_candidate(corrected[i2], frm.targets[i2],
                                             frm.num_targets[i2], xa12, ya12, xb12, yb12,
                                             frm.targets[i1][pt1].n, frm.targets[i1][pt1].nx,
                                             frm.targets[i1][pt1].ny, frm.targets[i1][pt1].sumg,
                                             vpar, cpar, calib[i2])

                if count > MAXCAND:
                    count = MAXCAND

                for j in range(count):
                    lists[i1][i2][i].p2[j] = cand[j].pnr
                    lists[i1][i2][i].corr[j] = cand[j].corr
                    lists[i1][i2][i].dist[j] = cand[j].tol
                lists[i1][i2][i].n = count

def take_best_candidates(
    src: List[NTupel],
    dst: List[NTupel],
    num_cams: int,
    num: int,
    tusage: np.ndarray
) -> int:
    quicksort_con(src)
    taken = 0

    for cand in range(num):
        has_used_target = False
        for cam in range(num_cams):
            tnum = src[cand].p[cam]
            if tnum > -1 and tusage[cam][tnum] > 0:
                has_used_target = True
                break

        if has_used_target:
            continue

        for cam in range(num_cams):
            tnum = src[cand].p[cam]
            if tnum > -1:
                tusage[cam][tnum] += 1
        dst[taken] = src[cand]
        taken += 1

    return taken

def correspondences(
    frm: Frame,
    corrected: List[List[Target]],
    vpar: VolumePar,
    cpar: ControlPar,
    calib: List[Calibration]
) -> tuple[list[NTupel], list[int]]:
    con0 = [NTupel([-1] * cpar.num_cams, 0.0) for _ in range(nmax)]
    con = [NTupel([-1] * cpar.num_cams, 0.0) for _ in range(nmax)]
    tim = safely_allocate_target_usage_marks(cpar.num_cams)
    if tim is None:
        print("out of memory")
        return [], []

    lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    if lists is None:
        print("list is not allocated")
        # deallocate_target_usage_marks(tim)
        return [], []

    match_counts = [0] * 4
    match_pairs(lists, corrected, frm, vpar, cpar, calib)

    if cpar.num_cams == 4:
        match0 = four_camera_matching(lists, frm.num_targets[0], vpar.corrmin, con0, 4 * nmax)
        match_counts[0] = take_best_candidates(con0, con, cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[0]

    if (cpar.num_cams == 4 and cpar.allCam_flag == 0) or cpar.num_cams == 3:
        match0 = three_camera_matching(lists, cpar.num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim)
        match_counts[1] = take_best_candidates(con0, con[match_counts[3]:], cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[1]

    if cpar.num_cams > 1 and cpar.allCam_flag == 0:
        match0 = consistent_pair_matching(lists, cpar.num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim)
        match_counts[2] = take_best_candidates(con0, con[match_counts[3]:], cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[2]

    for i in range(match_counts[3]):
        for j in range(cpar.num_cams):
            if con[i].p[j] < 0:
                continue
            p1 = corrected[j][con[i].p[j]].pnr
            if p1 > -1 and p1 < 1202590843:
                frm.targets[j][p1].tnr = i

    # deallocate_adjacency_lists(lists)
    # deallocate_target_usage_marks(tim)

    return con, match_counts
