from typing import List
import numpy as np
from pyoptv.epi import Coord2D
import pytest
from pyoptv.correspondences import (
    qs_target_y,
    quicksort_target_y,
    quicksort_coord2d_x,
    quicksort_con,
    match_pairs,
    safely_allocate_adjacency_lists,
    safely_allocate_target_usage_marks,
    four_camera_matching,
    three_camera_matching,
    consistent_pair_matching,
    correspondences,
    Correspond,
)
from pyoptv.parameters import ControlPar, read_control_par, read_volume_par
from pyoptv.calibration import Calibration, read_ori
from pyoptv.imgcoord import img_coord
from pyoptv.trafo import metric_to_pixel, pixel_to_metric, dist_to_flat
from pathlib import Path
from pyoptv.tracking_frame_buf import Frame, Target


@pytest.fixture(scope="session")
def testing_fodder_dir():
    """Fixture to provide the path to the testing_fodder directory inside pyoptv/tests."""
    return Path(__file__).parent / "testing_fodder"

# --- Sorting and utility tests ---
def test_qs_target_y():

    test_pix = [
        Target(0, 0.0, -0.2, 5, 1, 2, 10, -999),
        Target(6, 0.2, 0.0, 10, 8, 1, 20, -999),
        Target(3, 0.2, 0.8, 10, 3, 3, 30, -999),
        Target(4, 0.4, -1.1, 10, 3, 3, 40, -999),
        Target(1, 0.7, -0.1, 10, 3, 3, 50, -999),
        Target(7, 1.2, 0.3, 10, 3, 3, 60, -999),
        Target(5, 10.4, 0.1, 10, 3, 3, 70, -999),
    ]
    qs_target_y(test_pix, 0, 6)
    assert abs(test_pix[0].y + 1.1) < 1e-6
    assert abs(test_pix[1].y + 0.2) < 1e-6
    assert abs(test_pix[6].y - 0.8) < 1e-6

def test_quicksort_target_y():
    class Target:
        def __init__(self, pnr, x, y, n, nx, ny, sumg, tnr):
            self.pnr = pnr; self.x = x; self.y = y; self.n = n; self.nx = nx; self.ny = ny; self.sumg = sumg; self.tnr = tnr
    test_pix = [
        Target(0, 0.0, -0.2, 5, 1, 2, 10, -999),
        Target(6, 0.2, 0.0, 10, 8, 1, 20, -999),
        Target(3, 0.2, 0.8, 10, 3, 3, 30, -999),
        Target(4, 0.4, -1.1, 10, 3, 3, 40, -999),
        Target(1, 0.7, -0.1, 10, 3, 3, 50, -999),
        Target(7, 1.2, 0.3, 10, 3, 3, 60, -999),
        Target(5, 10.4, 0.1, 10, 3, 3, 70, -999),
    ]
    num = len(test_pix)
    quicksort_target_y(test_pix)
    assert abs(test_pix[0].y + 1.1) < 1e-6
    assert abs(test_pix[1].y + 0.2) < 1e-6
    assert abs(test_pix[num-1].y - 0.8) < 1e-6

def test_quicksort_coord2d_x():
    class Coord2D:
        def __init__(self, pnr, x, y):
            self.pnr = pnr; self.x = x; self.y = y
    test_crd = [
        Coord2D(0, 0.0, 0.0),
        Coord2D(6, 0.1, 0.1),
        Coord2D(3, 0.2, -0.8),
        Coord2D(4, -0.4, -1.1),
        Coord2D(1, 0.7, -0.1),
        Coord2D(7, 1.2, 0.3),
        Coord2D(5, 10.4, 0.1),
    ]
    num = len(test_crd)
    quicksort_coord2d_x(test_crd)
    assert abs(test_crd[0].x + 0.4) < 1e-6
    assert abs(test_crd[1].x - 0.0) < 1e-6
    assert abs(test_crd[num-1].x - 10.4) < 1e-6

def test_quicksort_con():
    class NTupel:
        def __init__(self, p, corr):
            self.p = p; self.corr = corr
    test_con = [
        NTupel([0, 1, 2, 3], 0.1),
        NTupel([0, 1, 2, 3], 0.2),
        NTupel([0, 1, 2, 3], 0.15),
    ]
    quicksort_con(test_con)
    assert abs(test_con[0].corr - 0.2) < 1e-6
    assert abs(test_con[2].corr - 0.1) < 1e-6


@pytest.fixture(scope="session")
def testing_fodder_dir():
    """Fixture to provide the path to the testing_fodder directory inside pyoptv/tests."""
    return Path(__file__).parent / "testing_fodder"

def read_all_calibration(cpar, testing_fodder_dir):
    ori_tmpl = testing_fodder_dir / "cal" / "sym_cam{}.tif.ori"
    added_name = testing_fodder_dir / "cal" / "cam1.tif.addpar"
    calib = [Calibration() for _ in range(cpar.num_cams)]
    for cam in range(cpar.num_cams):
        ori_name = str(ori_tmpl).format(cam + 1)
        calib[cam] = read_ori(ori_name, str(added_name))

    return calib

# Helper to generate a synthetic test set as in the C code
def generate_test_set(calib, cpar, vpar):
    frm = Frame(cpar.num_cams)
    frm.num_targets = [16 for _ in range(cpar.num_cams)]
    frm.targets = [[Target() for _ in range(16)] for _ in range(cpar.num_cams)]
    for cam in range(cpar.num_cams):
        for cpt_horz in range(4):
            for cpt_vert in range(4):
                cpt_ix = cpt_horz * 4 + cpt_vert
                if cam % 2:
                    cpt_ix = 15 - cpt_ix
                targ = frm.targets[cam][cpt_ix]
                targ.pnr = cpt_ix
                tmp = np.array([cpt_vert * 10, cpt_horz * 10, 0.0])
                # Store pixel coordinates (not metric) in Target.x/y
                x_metric, y_metric = img_coord(tmp, calib[cam], cpar.mm)
                x_pix, y_pix = metric_to_pixel(x_metric, y_metric, cpar)
                targ.x = x_pix
                targ.y = y_pix
                targ.n = 25
                targ.nx = 5
                targ.ny = 5
                targ.sumg = 10
    return frm

def correct_frame(frm: Frame, calib: List[Calibration], cpar: ControlPar, tol: float) -> List[List[Coord2D]]:
    corrected = []
    for cam in range(cpar.num_cams):
        cam_corr = []
        for part in range(frm.num_targets[cam]):
            c2d = Coord2D()
            c2d.x, c2d.y = pixel_to_metric(frm.targets[cam][part].x, frm.targets[cam][part].y, cpar)
            c2d.x, c2d.y = dist_to_flat(c2d.x, c2d.y, calib[cam], tol)
            c2d.pnr = frm.targets[cam][part].pnr
            cam_corr.append(c2d)
        quicksort_coord2d_x(cam_corr)
        corrected.append(cam_corr)
    return corrected

# --- Full correspondence and matching tests ---
def test_pairwise_matching(testing_fodder_dir):
    cpar = read_control_par(str(testing_fodder_dir / "parameters" / "ptv.par"))
    vpar = read_volume_par(str(testing_fodder_dir / "parameters" / "criteria.par"))
    cpar.mm.n2[0] = 1.0001
    cpar.mm.n3 = 1.0001
    calib = read_all_calibration(cpar, testing_fodder_dir)
    frm = generate_test_set(calib, cpar, vpar)
    corrected = correct_frame(frm, calib, cpar, 0.0001)
    lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    match_pairs(lists, corrected, frm, vpar, cpar, calib)

    # Deep check: for every cam pair and every target, check the candidate list matches ground truth
    for cam in range(cpar.num_cams - 1):
        for subcam in range(cam + 1, cpar.num_cams):
            for part in range(frm.num_targets[cam]):
                # Compute the expected ground truth candidate for this cam/part/subcam
                # The synthetic grid is 4x4, so pnr = cpt_ix = cpt_horz*4 + cpt_vert
                # For odd cameras, the index is reversed
                if (subcam % 2) == 0:
                    expected_pnr = part
                else:
                    expected_pnr = 15 - part
                # Find the index in corrected[subcam] with pnr == expected_pnr
                expected_idx = None
                for idx, c2d in enumerate(corrected[subcam]):
                    if c2d.pnr == expected_pnr:
                        expected_idx = idx
                        break
                assert expected_idx is not None, f"Expected pnr {expected_pnr} not found in cam {subcam}"
                # Now check that this index is present in the candidate list
                candidate_indices = lists[cam][subcam][part].p2
                assert expected_idx in candidate_indices, (
                    f"For cam {cam}, subcam {subcam}, part {part}: expected candidate idx {expected_idx} (pnr {expected_pnr}) not in candidates {candidate_indices}"
                )
                # There should be exactly one candidate (the ground truth)
                assert len(candidate_indices) == 1, (
                    f"For cam {cam}, subcam {subcam}, part {part}: expected 1 candidate, got {len(candidate_indices)}: {candidate_indices}"
                )

def test_four_camera_matching(testing_fodder_dir):
    cpar = read_control_par(str(testing_fodder_dir / "parameters" / "ptv.par"))
    vpar = read_volume_par(str(testing_fodder_dir / "parameters" / "criteria.par"))
    cpar.mm.n2[0] = 1.0001
    cpar.mm.n3 = 1.0001
    
    calib = read_all_calibration(cpar, testing_fodder_dir)
    frm = generate_test_set(calib, cpar, vpar)
    corrected = correct_frame(frm, calib, cpar, 0.0001)
    lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    match_pairs(lists, corrected, frm, vpar, cpar, calib)
    from pyoptv.correspondences import NTupel
    con = [NTupel() for _ in range(16)]
    matched = four_camera_matching(lists, 16, 1.0, con, 16)
    assert matched == 16

def test_three_camera_matching(testing_fodder_dir):
    cpar = read_control_par(str(testing_fodder_dir / "parameters" / "ptv.par"))
    vpar = read_volume_par(str(testing_fodder_dir / "parameters" / "criteria.par"))
    cpar.mm.n2[0] = 1.0001
    cpar.mm.n3 = 1.0001
    calib = read_all_calibration(cpar, testing_fodder_dir)
    frm = generate_test_set(calib, cpar, vpar)
    for part in range(frm.num_targets[1]):
        targ = frm.targets[1][part]
        targ.n = 0; targ.nx = 0; targ.ny = 0; targ.sumg = 0
    corrected = correct_frame(frm, calib, cpar, 0.0001)
    lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    match_pairs(lists, corrected, frm, vpar, cpar, calib)
    from pyoptv.correspondences import NTupel
    con = [NTupel() for _ in range(4*16)]
    tusage = safely_allocate_target_usage_marks(cpar.num_cams)
    matched = three_camera_matching(lists, 4, frm.num_targets, 100000.0, con, 4*16, tusage)
    assert matched == 16


def test_two_camera_matching(testing_fodder_dir):
    cpar = read_control_par(str(testing_fodder_dir / "parameters" / "ptv.par"))
    vpar = read_volume_par(str(testing_fodder_dir / "parameters" / "criteria.par"))
    cpar.mm.n2[0] = 1.0001
    cpar.mm.n3 = 1.0001
    vpar.Zmin_lay[0] = -1
    vpar.Zmin_lay[1] = -1
    vpar.Zmax_lay[0] = 1
    vpar.Zmax_lay[1] = 1
    calib = read_all_calibration(cpar, testing_fodder_dir)
    frm = generate_test_set(calib, cpar, vpar)
    cpar.num_cams = 2
    corrected = correct_frame(frm, calib, cpar, 0.0001)
    lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    match_pairs(lists, corrected, frm, vpar, cpar, calib)
    from pyoptv.correspondences import NTupel
    con = [NTupel() for _ in range(4*16)]
    tusage = safely_allocate_target_usage_marks(cpar.num_cams)
    matched = consistent_pair_matching(lists, 2, frm.num_targets, 10000.0, con, 4*16, tusage)
    assert matched == 16


def test_correspondences(testing_fodder_dir):
    cpar = read_control_par(str(testing_fodder_dir / "parameters" / "ptv.par"))
    vpar = read_volume_par(str(testing_fodder_dir / "parameters" / "criteria.par"))
    cpar.mm.n2[0] = 1.0001
    cpar.mm.n3 = 1.0001
    calib = read_all_calibration(cpar, testing_fodder_dir)
    frm = generate_test_set(calib, cpar, vpar)
    corrected = correct_frame(frm, calib, cpar, 0.0001)

    con, match_counts = correspondences(frm, corrected, vpar, cpar, calib)
    assert match_counts[0] == 16
    assert match_counts[1] == 0
    assert match_counts[2] == 0
    assert match_counts[3] == 16

    def test_quicksort_con_with_correspond():
        # Create Correspond objects with varying corr values
        c1 = Correspond(p1=0, n=0)
        c1.corr[0] = 0.5
        c1.n = 1
        c2 = Correspond(p1=1, n=0)
        c2.corr[0] = 0.8
        c2.n = 1
        c3 = Correspond(p1=2, n=0)
        c3.corr[0] = 0.3
        c3.n = 1

        # Set .corr attribute for sorting (simulate as in NTupel)
        c1.corr = 0.5
        c2.corr = 0.8
        c3.corr = 0.3

        con_list = [c1, c2, c3]
        quicksort_con(con_list)
        # Should be sorted descending by .corr
        assert con_list[0].corr == 0.8
        assert con_list[1].corr == 0.5
        assert con_list[2].corr == 0.3

    def test_quicksort_con_empty():
        con_list = []
        quicksort_con(con_list)  # Should not raise
        assert con_list == []

    def test_quicksort_con_single_element():
        c = Correspond(p1=0, n=0)
        c.corr = 1.23
        con_list = [c]
        quicksort_con(con_list)
        assert con_list[0].corr == 1.23


def test_minimal_pairwise_matching():
    from pyoptv.parameters import ControlPar, VolumePar
    from pyoptv.calibration import Calibration
    from pyoptv.tracking_frame_buf import Frame, Target
    from pyoptv.correspondences import match_pairs
    import numpy as np

    # Minimal control and volume parameters
    cpar = ControlPar(2)
    cpar.set_image_size((1000, 1000))
    cpar.set_pixel_size((1.0, 1.0))  # 1mm per pixel
    cpar.mm.n1 = 1.0
    cpar.mm.n2[0] = 1.0
    cpar.mm.n3 = 1.0

    vpar = VolumePar()
    vpar.X_lay = [-100, 100]
    vpar.Zmin_lay = [-100, 100]
    vpar.Zmax_lay = [-100, 100]
    vpar.cnx = 0.0
    vpar.cny = 0.0
    vpar.cn = 0.0
    vpar.csumg = 0.0
    vpar.corrmin = 0.0
    vpar.eps0 = 1.0  # small positive value for epipolar band

    # Two simple calibrations: identity (no rotation/translation)
    cal1 = Calibration()
    cal2 = Calibration()
    for cal in [cal1, cal2]:
        cal.ext_par = np.zeros(6)
        cal.int_par = np.zeros(7)
        cal.added_par = np.zeros(5)
        cal.glass_par = np.zeros(4)
        cal.principal_point = (500, 500)
        cal.cc = np.array([0.0, 0.0, 0.0])
        cal.ang = np.array([0.0, 0.0, 0.0])
        cal.k1 = 0.0
        cal.k2 = 0.0
        cal.k3 = 0.0
        cal.p1 = 0.0
        cal.p2 = 0.0
        cal.f = 1000.0
        cal.xh = 1000
        cal.yh = 1000
        cal.mmpx = 1.0
        cal.mmpy = 1.0

    # Two targets in world space
    world_targets = [np.array([10.0, 20.0, 30.0]), np.array([-10.0, -20.0, 30.0])]

    # Project to both cameras (no distortion, identity)
    frames = []
    for cal in [cal1, cal2]:
        frame = Frame(1)
        for i, X in enumerate(world_targets):
            # Simple pinhole: x = f*X/Z + cx, y = f*Y/Z + cy
            x = cal.f * X[0] / X[2] + cal.principal_point[0]
            y = cal.f * X[1] / X[2] + cal.principal_point[1]
            t = Target()
            t.pnr = i
            t.x = x
            t.y = y
            frame.targets.append(t)
        frames.append(frame)

    # Run pairwise matching (cam1 vs cam2)
    pairs = match_pairs(frames[0].targets, frames[1].targets, cal1, cal2, cpar, vpar)

    # There should be two pairs, each matching the same pnr
    assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}: {pairs}"
    pnrs_0 = [p[0] for p in pairs]
    pnrs_1 = [p[1] for p in pairs]
    assert set(pnrs_0) == {0, 1}, f"Unexpected pnr0: {pnrs_0}"
    assert set(pnrs_1) == {0, 1}, f"Unexpected pnr1: {pnrs_1}"
    # Each pair should match the same pnr
    for p0, p1 in pairs:
        assert p0 == p1, f"Pair mismatch: {p0} != {p1}"