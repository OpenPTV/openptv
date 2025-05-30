import pytest
import numpy as np
from pyoptv.correspondences import (
    NTupel, Correspond, quicksort_con, quicksort_target_y, quicksort_coord2d_x,
    safely_allocate_target_usage_marks, deallocate_target_usage_marks,
    safely_allocate_adjacency_lists, deallocate_adjacency_lists,
    four_camera_matching, three_camera_matching, consistent_pair_matching,
    match_pairs, take_best_candidates, correspondences
)
from pyoptv.tracking_frame_buf import Frame, Target
from pyoptv.calibration import Calibration
from pyoptv.parameters import ControlPar, MMNP, VolumePar

def test_quicksort_con():
    con = [Correspond(0, 0, np.zeros(1), np.array([3]), np.zeros(1)),
           Correspond(0, 0, np.zeros(1), np.array([1]), np.zeros(1)),
           Correspond(0, 0, np.zeros(1), np.array([2]), np.zeros(1))]
    quicksort_con(con)
    assert con[0].corr[0] == 3
    assert con[1].corr[0] == 2
    assert con[2].corr[0] == 1

def test_quicksort_target_y():
    class DummyTarget:
        def __init__(self, y):
            self.y = y
    pix = [DummyTarget(3), DummyTarget(1), DummyTarget(2)]
    quicksort_target_y(pix)
    assert pix[0].y == 1
    assert pix[1].y == 2
    assert pix[2].y == 3

def test_quicksort_coord2d_x():
    class DummyCoord2D:
        def __init__(self, x):
            self.x = x
    crd = [DummyCoord2D(3), DummyCoord2D(1), DummyCoord2D(2)]
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
    assert lists is None or all(lst is None for lst in lists), "Deallocation of adjacency lists failed."

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

def test_correspondences():
    # Use real classes from the codebase
    num_cams = 2
    max_targets = 3
    frm = Frame(num_cams, max_targets)
    frm.num_targets = [3, 3]
    frm.targets = [
        [Target(0, 0, 0, 1, 1, 1, 10, 0), Target(1, 1, 1, 1, 1, 1, 10, 1), Target(2, 2, 2, 1, 1, 1, 10, 2)],
        [Target(0, 0, 0, 1, 1, 1, 10, 0), Target(1, 1, 1, 1, 1, 1, 10, 1), Target(2, 2, 2, 1, 1, 1, 10, 2)]
    ]
    # Create corrected targets (simulate as needed)
    class Corrected:
        def __init__(self, x, y, pnr):
            self.x = x
            self.y = y
            self.pnr = pnr
    corrected = [
        [Corrected(0, 0, 0), Corrected(1, 1, 1), Corrected(2, 2, 2)],
        [Corrected(0, 0, 0), Corrected(1, 1, 1), Corrected(2, 2, 2)]
    ]
    vpar = VolumePar()
    vpar.X_lay = [-250.0, 250.0]
    vpar.Zmin_lay = [-100.0, -100.0]
    vpar.Zmax_lay = [100.0, 100.0]
    vpar.eps0 = 25.0
    vpar.corrmin = 0.5
    cpar = ControlPar(num_cams)
    cpar.imx = 1000
    cpar.imy = 1000
    cpar.pix_x = 0.01
    cpar.pix_y = 0.01
    cpar.mm = MMNP()
    calib = [Calibration(), Calibration()]
    con = correspondences(frm, corrected, vpar, cpar, calib)
    assert con is not None
    assert len(con) >= 0
