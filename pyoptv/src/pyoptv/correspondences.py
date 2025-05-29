import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .epi import epi_mm, find_candidate

nmax = 202400
MAXCAND = 100

class NTupel:
    def __init__(self, p, corr):
        self.p = p
        self.corr = corr

class Correspond:
    def __init__(self, p1, n, p2, corr, dist):
        self.p1 = p1
        self.n = n
        self.p2 = p2
        self.corr = corr
        self.dist = dist

def quicksort_con(con):
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

def quicksort_target_y(pix):
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

def quicksort_coord2d_x(crd):
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

def safely_allocate_target_usage_marks(num_cams):
    try:
        tusage = np.zeros((num_cams, nmax), dtype=np.int32)
        return tusage
    except MemoryError:
        return None

def deallocate_target_usage_marks(tusage):
    del tusage

def safely_allocate_adjacency_lists(num_cams, target_counts):
    try:
        lists = [[None for _ in range(num_cams)] for _ in range(num_cams)]
        for c1 in range(num_cams - 1):
            for c2 in range(c1 + 1, num_cams):
                lists[c1][c2] = [Correspond(0, 0, np.zeros(MAXCAND, dtype=np.int32), np.zeros(MAXCAND), np.zeros(MAXCAND)) for _ in range(target_counts[c1])]
        return lists
    except MemoryError:
        return None

def deallocate_adjacency_lists(lists):
    del lists

def four_camera_matching(lists, base_target_count, accept_corr, scratch, scratch_size):
    matched = 0
    for i in range(base_target_count):
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

                                if corr <= accept_corr:
                                    continue

                                scratch[matched] = NTupel([p1, p2, p3, p4], corr)
                                matched += 1
                                if matched == scratch_size:
                                    print("Overflow in correspondences.")
                                    return matched
    return matched

def three_camera_matching(lists, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage):
    matched = 0
    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
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

                                if corr <= accept_corr:
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

def consistent_pair_matching(lists, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage):
    matched = 0
    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = lists[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                if lists[i1][i2][i].n != 1:
                    continue

                p2 = lists[i1][i2][i].p2[0]
                if p2 > nmax or tusage[i2][p2] > 0:
                    continue

                corr = lists[i1][i2][i].corr[0] / lists[i1][i2][i].dist[0]
                if corr <= accept_corr:
                    continue

                scratch[matched] = NTupel([-2] * num_cams, corr)
                scratch[matched].p[i1] = p1
                scratch[matched].p[i2] = p2
                matched += 1
                if matched == scratch_size:
                    print("Overflow in correspondences.")
                    return matched
    return matched

def match_pairs(lists, corrected, frm, vpar, cpar, calib):
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

def take_best_candidates(src, dst, num_cams, num_cands, tusage):
    quicksort_con(src)
    taken = 0

    for cand in range(num_cands):
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

def correspondences(frm, corrected, vpar, cpar, calib):
    con0 = [NTupel([-1] * cpar.num_cams, 0.0) for _ in range(nmax)]
    con = [NTupel([-1] * cpar.num_cams, 0.0) for _ in range(nmax)]
    tim = safely_allocate_target_usage_marks(cpar.num_cams)
    if tim is None:
        print("out of memory")
        return None

    lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    if lists is None:
        print("list is not allocated")
        deallocate_target_usage_marks(tim)
        return None

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

    deallocate_adjacency_lists(lists)
    deallocate_target_usage_marks(tim)

    return con
