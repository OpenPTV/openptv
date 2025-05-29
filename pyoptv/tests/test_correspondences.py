import pytest
import numpy as np
from pyoptv.correspondences import (
    NTupel, Correspond, quicksort_con, quicksort_target_y, quicksort_coord2d_x,
    safely_allocate_target_usage_marks, deallocate_target_usage_marks,
    safely_allocate_adjacency_lists, deallocate_adjacency_lists,
    four_camera_matching, three_camera_matching, consistent_pair_matching,
    match_pairs, take_best_candidates, correspondences
)

def test_quicksort_con():
    con = [Correspond(0, 0, np.zeros(1), np.array([3]), np.zeros(1)),
           Correspond(0, 0, np.zeros(1), np.array([1]), np.zeros(1)),
           Correspond(0, 0, np.zeros(1), np.array([2]), np.zeros(1))]
    quicksort_con(con)
    assert con[0].corr[0] == 3
    assert con[1].corr[0] == 2
    assert con[2].corr[0] == 1

def test_quicksort_target_y():
    class Target:
        def __init__(self, y):
            self.y = y

    pix = [Target(3), Target(1), Target(2)]
    quicksort_target_y(pix)
    assert pix[0].y == 1
    assert pix[1].y == 2
    assert pix[2].y == 3

def test_quicksort_coord2d_x():
    class Coord2D:
        def __init__(self, x):
            self.x = x

    crd = [Coord2D(3), Coord2D(1), Coord2D(2)]
    quicksort_coord2d_x(crd)
    assert crd[0].x == 1
    assert crd[1].x == 2
    assert crd[2].x == 3

def test_safely_allocate_target_usage_marks():
    tusage = safely_allocate_target_usage_marks(2)
    assert tusage.shape == (2, 202400)
    deallocate_target_usage_marks(tusage)

def test_safely_allocate_adjacency_lists():
    lists = safely_allocate_adjacency_lists(2, [3, 3])
    assert len(lists) == 2
    assert len(lists[0]) == 2
    assert len(lists[0][1]) == 3
    deallocate_adjacency_lists(lists)

def test_four_camera_matching():
    lists = safely_allocate_adjacency_lists(4, [3, 3, 3, 3])
    for i in range(3):
        lists[0][1][i].n = 1
        lists[0][1][i].p2[0] = i
        lists[0][1][i].corr[0] = 1.0
        lists[0][1][i].dist[0] = 1.0
        lists[0][2][i].n = 1
        lists[0][2][i].p2[0] = i
        lists[0][2][i].corr[0] = 1.0
        lists[0][2][i].dist[0] = 1.0
        lists[0][3][i].n = 1
        lists[0][3][i].p2[0] = i
        lists[0][3][i].corr[0] = 1.0
        lists[0][3][i].dist[0] = 1.0
        lists[1][2][i].n = 1
        lists[1][2][i].p2[0] = i
        lists[1][2][i].corr[0] = 1.0
        lists[1][2][i].dist[0] = 1.0
        lists[1][3][i].n = 1
        lists[1][3][i].p2[0] = i
        lists[1][3][i].corr[0] = 1.0
        lists[1][3][i].dist[0] = 1.0
        lists[2][3][i].n = 1
        lists[2][3][i].p2[0] = i
        lists[2][3][i].corr[0] = 1.0
        lists[2][3][i].dist[0] = 1.0

    scratch = [NTupel([0, 0, 0, 0], 0.0) for _ in range(4 * 202400)]
    matched = four_camera_matching(lists, 3, 0.5, scratch, 4 * 202400)
    assert matched == 3
    deallocate_adjacency_lists(lists)

def test_three_camera_matching():
    lists = safely_allocate_adjacency_lists(3, [3, 3, 3])
    for i in range(3):
        lists[0][1][i].n = 1
        lists[0][1][i].p2[0] = i
        lists[0][1][i].corr[0] = 1.0
        lists[0][1][i].dist[0] = 1.0
        lists[0][2][i].n = 1
        lists[0][2][i].p2[0] = i
        lists[0][2][i].corr[0] = 1.0
        lists[0][2][i].dist[0] = 1.0
        lists[1][2][i].n = 1
        lists[1][2][i].p2[0] = i
        lists[1][2][i].corr[0] = 1.0
        lists[1][2][i].dist[0] = 1.0

    scratch = [NTupel([0, 0, 0], 0.0) for _ in range(4 * 202400)]
    tusage = safely_allocate_target_usage_marks(3)
    matched = three_camera_matching(lists, 3, [3, 3, 3], 0.5, scratch, 4 * 202400, tusage)
    assert matched == 3
    deallocate_adjacency_lists(lists)
    deallocate_target_usage_marks(tusage)

def test_consistent_pair_matching():
    lists = safely_allocate_adjacency_lists(2, [3, 3])
    for i in range(3):
        lists[0][1][i].n = 1
        lists[0][1][i].p2[0] = i
        lists[0][1][i].corr[0] = 1.0
        lists[0][1][i].dist[0] = 1.0

    scratch = [NTupel([0, 0], 0.0) for _ in range(4 * 202400)]
    tusage = safely_allocate_target_usage_marks(2)
    matched = consistent_pair_matching(lists, 2, [3, 3], 0.5, scratch, 4 * 202400, tusage)
    assert matched == 3
    deallocate_adjacency_lists(lists)
    deallocate_target_usage_marks(tusage)

def test_match_pairs():
    class Target:
        def __init__(self, x, y, pnr):
            self.x = x
            self.y = y
            self.pnr = pnr
            self.n = 10  # number of pixels in target
            self.nx = 3  # size in x direction
            self.ny = 3  # size in y direction
            self.sumg = 100  # sum of grey values

    class Frame:
        def __init__(self):
            self.num_targets = [3, 3]
            self.targets = [[Target(0, 0, 0), Target(1, 1, 1), Target(2, 2, 2)],
                            [Target(0, 0, 0), Target(1, 1, 1), Target(2, 2, 2)]]

    class Calibration:
        def __init__(self):
            self.int_par = self.InteriorParams()

        class InteriorParams:
            def __init__(self):
                self.xh = 0.0
                self.yh = 0.0 # Though yh is not used in the failing line, it's good practice to add it

    class ControlParams:
        def __init__(self):
            self.num_cams = 2
            self.mm = MultimediaParams()  # Add multimedia parameters
            self.pix_x = 0.01 # Added pix_x
            self.pix_y = 0.01 # Added pix_y for completeness
            self.imx = 1280 # Added imx
            self.imy = 1024 # Added imy for completeness

    class MultimediaParams:
        pass

    class VolumeParams:
        def __init__(self):
            self.X_lay = [-250.0, 250.0]  # X boundaries for the measurement volume
            self.Zmin_lay = [-100.0, -100.0]  # minimum Z values
            self.Zmax_lay = [100.0, 100.0]  # maximum Z values
            self.eps0 = 25.0  # epipolar band width
            self.corrmin = 0.5  # minimum correlation

    class Corrected:
        def __init__(self, x, y, pnr):
            self.x = x
            self.y = y
            self.pnr = pnr

    frm = Frame()
    corrected = [[Corrected(0, 0, 0), Corrected(1, 1, 1), Corrected(2, 2, 2)],
                 [Corrected(0, 0, 0), Corrected(1, 1, 1), Corrected(2, 2, 2)]]
    vpar = VolumeParams()
    cpar = ControlParams()
    calib = [Calibration(), Calibration()]

    lists = safely_allocate_adjacency_lists(2, [3, 3])
    match_pairs(lists, corrected, frm, vpar, cpar, calib)
    assert lists[0][1][0].n == 1
    assert lists[0][1][1].n == 1
    assert lists[0][1][2].n == 1
    deallocate_adjacency_lists(lists)

def test_take_best_candidates():
    src = [NTupel([0, 0], 1.0), NTupel([1, 1], 0.5), NTupel([2, 2], 0.75)]
    dst = [NTupel([0, 0], 0.0) for _ in range(3)]
    tusage = safely_allocate_target_usage_marks(2)
    taken = take_best_candidates(src, dst, 2, 3, tusage)
    assert taken == 3
    assert dst[0].corr == 1.0
    assert dst[1].corr == 0.75
    assert dst[2].corr == 0.5
    deallocate_target_usage_marks(tusage)

def test_correspondences():
    class Target:
        def __init__(self, x, y, pnr):
            self.x = x
            self.y = y
            self.pnr = pnr
            self.n = 10  # number of pixels in target
            self.nx = 3  # size in x direction
            self.ny = 3  # size in y direction  
            self.sumg = 100  # sum of grey values

    class Frame:
        def __init__(self):
            self.num_targets = [3, 3]
            self.targets = [[Target(0, 0, 0), Target(1, 1, 1), Target(2, 2, 2)],
                            [Target(0, 0, 0), Target(1, 1, 1), Target(2, 2, 2)]]

    class Calibration:
        pass

    class ControlParams:
        def __init__(self):
            self.num_cams = 2
            self.mm = MultimediaParams()  # Add multimedia parameters

    class MultimediaParams:
        pass

    class VolumeParams:
        def __init__(self):
            self.X_lay = [-250.0, 250.0]  # X boundaries for the measurement volume
            self.Zmin_lay = [-100.0, -100.0]  # minimum Z values
            self.Zmax_lay = [100.0, 100.0]  # maximum Z values
            self.eps0 = 25.0  # epipolar band width
            self.corrmin = 0.5  # minimum correlation

    class Corrected:
        def __init__(self, x, y, pnr):
            self.x = x
            self.y = y
            self.pnr = pnr

    frm = Frame()
    corrected = [[Corrected(0, 0, 0), Corrected(1, 1, 1), Corrected(2, 2, 2)],
                 [Corrected(0, 0, 0), Corrected(1, 1, 1), Corrected(2, 2, 2)]]
    vpar = VolumeParams()
    cpar = ControlParams()
    calib = [Calibration(), Calibration()]

    con = correspondences(frm, corrected, vpar, cpar, calib)
    assert con is not None
    assert len(con) > 0
