
from typing import List, Tuple

PT_UNUSED = -999

class Correspond:
    def __init__(self, n: int, p1: int):
        self.n = n
        self.p1 = p1

def quicksort_con(con, num):
    if num > 0:
        qs_con(con, 0, num-1)

def qs_con(con, left, right):
    if left >= right:
        return

    pivot = con[(left+right)//2].corr
    i, j = left, right
    while i <= j:
        while con[i].corr < pivot:
            i += 1
        while con[j].corr > pivot:
            j -= 1
        if i <= j:
            con[i], con[j] = con[j], con[i]
            i += 1
            j -= 1

    qs_con(con, left, j)
    qs_con(con, i, right)


def quicksort_target_y(pix, num):
    qs_target_y(pix, 0, num-1)

def qs_target_y(pix, left, right):
    if left >= right:
        return

    pivot = pix[(left+right)//2].y
    i, j = left, right
    while i <= j:
        while pix[i].y < pivot:
            i += 1
        while pix[j].y > pivot:
            j -= 1
        if i <= j:
            pix[i], pix[j] = pix[j], pix[i]
            i += 1
            j -= 1

    qs_target_y(pix, left, j)
    qs_target_y(pix, i, right)

def quicksort_coord2d_x(crd, num):
    qs_coord2d_x(crd, 0, num-1)

def qs_coord2d_x(crd, left, right):
    if left >= right:
        return

    pivot = crd[(left+right)//2].x
    i, j = left, right
    while i <= j:
        while crd[i].x < pivot and i < right:
            i += 1
        while pivot < crd[j].x and j > left:
            j -= 1
        if i <= j:
            crd[i], crd[j] = crd[j], crd[i]
            i += 1
            j -= 1

    qs_coord2d_x(crd, left, j)
    qs_coord2d_x(crd, i, right)


def deallocate_target_usage_marks(tusage, num_cams):
    for cam in range(num_cams):
        del tusage[cam][:]
    del tusage


def safely_allocate_target_usage_marks(num_cams, nmax):
    tusage = []
    error = False
    
    try:
        for cam in range(num_cams):
            tusage_cam = [0] * nmax
            tusage.append(tusage_cam)
    except MemoryError:
        error = True
    
    if error:
        deallocate_target_usage_marks(tusage, num_cams)
        return None
    else:
        return tusage


def deallocate_adjacency_lists(lists: List[List[List[correspond]]], num_cams: int) -> None:
    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            lists[c1][c2] = None
            lists[c2][c1] = None
            del lists[c1][c2]
            del lists[c2][c1]



def deallocate_adjacency_lists(lists: List[List[List[Correspond]]], num_cams: int) -> None:
    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            for i in range(len(lists[c1][c2])):
                lists[c1][c2][i] = None
            lists[c1][c2] = None

def safely_allocate_adjacency_lists(lists: List[List[List[Correspond]]], num_cams: int, target_counts: List[int]) -> int:
    error = False

    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            if not error:
                try:
                    lists[c1][c2] = [Correspond(0, 0) for i in range(target_counts[c1])]
                except MemoryError:
                    error = True
                    continue
                
                for edge in range(target_counts[c1]):
                    lists[c1][c2][edge].n = 0
                    lists[c1][c2][edge].p1 = 0
            else:
                lists[c1][c2] = None

    if error:
        deallocate_adjacency_lists(lists, num_cams)
        return 0
    else:
        return 1




def four_camera_matching(list, base_target_count, accept_corr, scratch, scratch_size):
    matched = 0

    for i in range(base_target_count):
        p1 = list[0][1][i].p1
        for j in range(list[0][1][i].n):
            p2 = list[0][1][i].p2[j]
            for k in range(list[0][2][i].n):
                p3 = list[0][2][i].p2[k]
                for l in range(list[0][3][i].n):
                    p4 = list[0][3][i].p2[l]

                    for m in range(list[1][2][p2].n):
                        p31 = list[1][2][p2].p2[m]
                        if p3 != p31:
                            continue

                        for n in range(list[1][3][p2].n):
                            p41 = list[1][3][p2].p2[n]
                            if p4 != p41:
                                continue

                            for o in range(list[2][3][p3].n):
                                p42 = list[2][3][p3].p2[o]
                                if p4 != p42:
                                    continue

                                corr = (list[0][1][i].corr[j]
                                        + list[0][2][i].corr[k]
                                        + list[0][3][i].corr[l]
                                        + list[1][2][p2].corr[m]
                                        + list[1][3][p2].corr[n]
                                        + list[2][3][p3].corr[o]) \
                                        / (list[0][1][i].dist[j]
                                        + list[0][2][i].dist[k]
                                        + list[0][3][i].dist[l]
                                        + list[1][2][p2].dist[m]
                                        + list[1][3][p2].dist[n]
                                        + list[2][3][p3].dist[o])

                                if corr <= accept_corr:
                                    continue

                                # accept as preliminary match
                                scratch[matched].p[0] = p1
                                scratch[matched].p[1] = p2
                                scratch[matched].p[2] = p3
                                scratch[matched].p[3] = p4
                                scratch[matched].corr = corr

                                matched += 1
                                if matched == scratch_size:
                                    print("Overflow in correspondences.")
                                    return matched

    return matched


import numpy as np

def three_camera_matching(list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage):
    matched = 0
    nmax = np.inf
    
    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = list[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue
                
                for j in range(list[i1][i2][i].n):
                    p2 = list[i1][i2][i].p2[j]
                    if p2 > nmax or tusage[i2][p2] > 0:
                        continue

                    for i3 in range(i2 + 1, num_cams):
                        for k in range(list[i1][i3][i].n):
                            p3 = list[i1][i3][i].p2[k]
                            if p3 > nmax or tusage[i3][p3] > 0:
                                continue
							
                            indices = np.where(list[i2][i3][p2].p2 == p3)[0]
                            if indices.size == 0:
                                continue
							
                            m = indices[0]
                            corr = (list[i1][i2][i].corr[j] + list[i1][i3][i].corr[k] + list[i2][i3][p2].corr[m]) / (list[i1][i2][i].dist[j] + list[i1][i3][i].dist[k] + list[i2][i3][p2].dist[m])
							
                            if corr <= accept_corr:
                                continue
							
                            p = np.full(num_cams, -2)
                            p[i1], p[i2], p[i3] = p1, p2, p3
                            scratch[matched].p = p
                            scratch[matched].corr = corr
							
                            matched += 1
                            if matched == scratch_size:
                                print ("Overflow in correspondences.\n")
                                return matched
    return matched


import numpy as np

def consistent_pair_matching(list, num_cams, target_counts, accept_corr, scratch, scratch_size, tusage):
    matched = 0
    nmax = np.inf
    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = list[i1][i2][i].p1
                if p1 > nmax or tusage[i1][p1] > 0:
                    continue

                if list[i1][i2][i].n != 1:
                    continue

                p2 = list[i1][i2][i].p2[0]
                if p2 > nmax or tusage[i2][p2] > 0:
                    continue

                corr = list[i1][i2][i].corr[0] / list[i1][i2][i].dist[0]
                if corr <= accept_corr:
                    continue

                for n in range(num_cams):
                    scratch[matched].p[n] = -2

                scratch[matched].p[i1] = p1
                scratch[matched].p[i2] = p2
                scratch[matched].corr = corr

                matched += 1
                if matched == scratch_size:
                    print("Overflow in correspondences.\n")
                    return matched

    return matched


def match_pairs(list, corrected, frm, vpar, cpar, calib):
    MAXCAND = 100
    for i1 in range(cpar.num_cams - 1):
        for i2 in range(i1 + 1, cpar.num_cams):
            for i in range(frm.num_targets[i1]):
                if corrected[i1][i].x == PT_UNUSED:
                    continue
                
                xa12, ya12, xb12, yb12 = epi_mm(corrected[i1][i].x, corrected[i1][i].y, 
                                                calib[i1], calib[i2], cpar.mm, 
                                                vpar)
                
                # origin point in the list
                list[i1][i2][i].p1 = i
                pt1 = corrected[i1][i].pnr
                
                # search for a conjugate point in corrected[i2]
                cand = [Candidate() for _ in range(MAXCAND)]
                count = find_candidate(corrected[i2], frm.targets[i2],
                                       frm.num_targets[i2], xa12, ya12, xb12, yb12, 
                                       frm.targets[i1][pt1].n, frm.targets[i1][pt1].nx,
                                       frm.targets[i1][pt1].ny, frm.targets[i1][pt1].sumg, cand, 
                                       vpar, cpar, calib[i2])
                
                # write all corresponding candidates to the preliminary list of correspondences
                if count > MAXCAND:
                    count = MAXCAND
                
                for j in range(count):
                    list[i1][i2][i].p2[j] = cand[j].pnr
                    list[i1][i2][i].corr[j] = cand[j].corr
                    list[i1][i2][i].dist[j] = cand[j].tol
                
                list[i1][i2][i].n = count


def take_best_candidates(src, dst, num_cams, num_cands, tusage):
    taken = 0

    # sort candidates by match quality (.corr)
    src.sort(key=lambda x: x.corr, reverse=True)

    # take quadruplets from the top to the bottom of the sorted list
    # only if none of the points has already been used
    for cand in src:
        has_used_target = False
        for cam in range(num_cams):
            tnum = cand.p[cam]

            # if any correspondence in this camera, check that target is free
            if tnum > -1 and tusage[cam][tnum] > 0:
                has_used_target = True
                break

        if has_used_target:
            continue

        # Only now can we commit to marking used targets.
        for cam in range(num_cams):
            tnum = cand.p[cam]
            if tnum > -1:
                tusage[cam][tnum] += 1

        dst[taken] = cand
        taken += 1

    return taken



def correspondences(frm: frame, corrected: List[List[coord_2d]], 
                    vpar: volume_par, cpar: control_par, 
                    calib: List[List[Calibration]], 
                    match_counts: List[int]) -> List[n_tupel]:

    nmax = 1000

    # Allocation of scratch buffers for internal tasks and return-value space
    con0 = (nmax * cpar.num_cams) * [n_tupel(p=[-1]*cpar.num_cams, corr=0.0)]
    con = (nmax * cpar.num_cams) * [n_tupel(p=[-1]*cpar.num_cams, corr=0.0)]

    tim = safely_allocate_target_usage_marks(cpar.num_cams)
    if tim is None:
        print("out of memory")
        return None

    # allocate memory for lists of correspondences
    list = [[None]*4 for _ in range(4)]
    if safely_allocate_adjacency_lists(list, cpar.num_cams, frm.num_targets) == 0:
        print("list is not allocated")
        deallocate_target_usage_marks(tim, cpar.num_cams)
        return None

    # if I understand correctly, the number of matches cannot be more than the number of
    # targets (dots) in the first image. In the future we'll replace it by the maximum
    # number of targets in any image (if we will implement the cyclic search) but for
    # a while we always start with the cam1
    for i in range(nmax):
        for j in range(cpar.num_cams):
            con0[i].p[j] = -1
        con0[i].corr = 0.0

    for i in range(4):
        match_counts[i] = 0

    # Generate adjacency lists: mark candidates for correspondence.
    # matching 1 -> 2,3,4 + 2 -> 3,4 + 3 -> 4
    match_pairs(list, corrected, frm, vpar, cpar, calib)

    # search consistent quadruplets in the list
    if cpar.num_cams == 4:
        match0 = four_camera_matching(list, frm.num_targets[0], 
                                      vpar.corrmin, con0, 4*nmax)

        match_counts[0] = take_best_candidates(con0, con, cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[0]

    # search consistent triplets: 123, 124, 134, 234
    if (cpar.num_cams == 4 and cpar.allCam_flag == 0) or cpar.num_cams == 3:
        match0 = three_camera_matching(list, cpar.num_cams, frm.num_targets, 
                                        vpar.corrmin, con0, 4*nmax, tim)

        match_counts[1] = take_best_candidates(con0, con[match_counts[3]:], 
                                                cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[1]

    # Search consistent pairs: 12, 13, 14, 23, 24, 34
    if cpar.num_cams > 1 and cpar.allCam_flag == 0:
        match0 = consistent_pair_matching(list, cpar.num_cams, frm.num_targets, 
                                           vpar.corrmin, con0, 4*nmax, tim)
        match_counts[2] = take_best_candidates(con0, con[match_counts[3]:], 
                                                cpar.num_cams, match0, tim)
        match_counts[3] += match_counts[2]
    
    # Give each used pix the correspondence number
    for i in range(match_counts[3]):
        for j in range(cpar.num_cams):
            # Skip cameras without a correspondence obviously.
            if con[i].p[j] < 0:
                continue
            
            p1 = corrected[j][con[i].p[j]].pnr
            if p1 > -1 and p1 < 1202590843:
                frm.targets[j][p1].tnr = i
    
    # Free all other allocations
    deallocate_adjacency_lists(list, cpar.num_cams)
    deallocate_target_usage_marks(tim, cpar.num_cams)
    free(con0)
    
    return con
