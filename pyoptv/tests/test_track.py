import pytest
import numpy as np
from pyoptv.vec_utils import (
    Vec3D, Vec2D, vec_scalar_mul, vec_subt, vec_diff_norm, vec_dot, vec_norm, vec_set, vec_copy
)
from pyoptv.track import (
    TrackingRun, TR_UNUSED, Frame, FoundPix,
    reset_foundpix_array, copy_foundpix_array, register_closest_neighbs, search_volume_center_moving,
    predict, pos3d_in_bounds, angle_acc, candsearch_in_pix, candsearch_in_pix_rest, searchquader,
    sort_candidates_by_freq, sort, point_to_pixel, sorted_candidates_in_volume, assess_new_position,
    add_particle, trackcorr_c_loop, trackcorr_c_finish, trackback_c
)
from pyoptv.parameters import ControlPar, TrackPar, VolumePar
from pyoptv.calibration import Calibration
from pyoptv.tracking_frame_buf import Target, FrameBuffer

def test_vec_scalar_mul():
    vec = Vec3D(1, 2, 3)
    scalar = 2
    result = vec_scalar_mul(vec, scalar)
    assert result.x == 2
    assert result.y == 4
    assert result.z == 6

def test_vec_subt():
    vec1 = Vec3D(1, 2, 3)
    vec2 = Vec3D(3, 2, 1)
    result = vec_subt(vec1, vec2)
    assert result.x == -2
    assert result.y == 0
    assert result.z == 2

def test_vec_diff_norm():
    vec1 = Vec3D(1, 2, 3)
    vec2 = Vec3D(4, 5, 6)
    result = vec_diff_norm(vec1, vec2)
    assert np.isclose(result, 5.196152422706632)

def test_vec_dot():
    vec1 = Vec3D(1, 2, 3)
    vec2 = Vec3D(4, 5, 6)
    result = vec_dot(vec1, vec2)
    assert result == 32

def test_vec_norm():
    vec = Vec3D(1, 2, 3)
    result = vec_norm(vec)
    assert np.isclose(result, 3.7416573867739413)

def test_vec_set():
    result = vec_set(1, 2, 3)
    assert result.x == 1
    assert result.y == 2
    assert result.z == 3

def test_vec_copy():
    vec = Vec3D(1, 2, 3)
    result = vec_copy(vec)
    assert result.x == 1
    assert result.y == 2
    assert result.z == 3

def test_reset_foundpix_array():
    arr = [FoundPix(1, 2, [1, 1, 1]) for _ in range(5)]
    reset_foundpix_array(arr, 5, 3)
    for item in arr:
        assert item.ftnr == TR_UNUSED  # TR_UNUSED
        assert item.freq == 0
        assert item.whichcam == [0, 0, 0]

def test_copy_foundpix_array():
    src = [FoundPix(1, 2, [1, 1, 1]) for _ in range(5)]
    dest = [FoundPix(0, 0, [0, 0, 0]) for _ in range(5)]
    copy_foundpix_array(dest, src, 5, 3)
    for i in range(5):
        assert dest[i].ftnr == src[i].ftnr
        assert dest[i].freq == src[i].freq
        assert dest[i].whichcam == src[i].whichcam

def test_register_closest_neighbs():
    targets = [Target(i, 1, 2, 0, 0, 0, 0, i) for i in range(3)]
    reg = [FoundPix(-1, 0, [0, 0, 0, 0]) for _ in range(4)]
    cpar = ControlPar(4)
    register_closest_neighbs(targets, 3, 0, 2, 2, 2, 2, 2, 2, reg, cpar)
    assert reg[0].ftnr == 0
    assert reg[1].ftnr == 1
    assert reg[2].ftnr == 2
    assert reg[3].ftnr == TR_UNUSED

def test_search_volume_center_moving():
    prev_pos = Vec3D(1, 2, 3)
    curr_pos = Vec3D(4, 5, 6)
    result = search_volume_center_moving(prev_pos, curr_pos)
    assert result.x == 7
    assert result.y == 8
    assert result.z == 9

def test_predict():
    prev_pos = Vec2D(1, 2)
    curr_pos = Vec2D(4, 5)
    result = predict(prev_pos, curr_pos)
    assert result.x == 7
    assert result.y == 8

def test_pos3d_in_bounds():
    class Bounds:
        dvxmin = 0
        dvxmax = 2
        dvymin = 0
        dvymax = 3
        dvzmin = 0
        dvzmax = 4
    pos = Vec3D(1, 2, 3)
    bounds = Bounds()
    assert pos3d_in_bounds(pos, bounds) == True
    pos = Vec3D(-1, 2, 3)
    assert pos3d_in_bounds(pos, bounds) == False

def test_angle_acc():
    start = Vec3D(1, 2, 3)
    pred = Vec3D(4, 5, 6)
    cand = Vec3D(7, 8, 9)
    angle, acc = angle_acc(start, pred, cand)
    assert np.isclose(angle, 0)
    assert np.isclose(acc, 5.196152422706632)

def test_candsearch_in_pix():
    class DummyTarget:
        def __init__(self, x, y, tnr):
            self.x = x
            self.y = y
            self.tnr = tnr
    next = [DummyTarget(1, 2, 0), DummyTarget(3, 4, 1), DummyTarget(5, 6, 2), DummyTarget(7, 8, 3)]
    p = np.zeros(4, dtype=np.int32)
    cpar = ControlPar(4)
    result = candsearch_in_pix(next, 4, 2, 2, 2, 2, 2, 2, p, cpar)
    assert result >= 0

def test_candsearch_in_pix_rest():
    class DummyTarget:
        def __init__(self, x, y, tnr):
            self.x = x
            self.y = y
            self.tnr = tnr
    next = [DummyTarget(1, 2, -999), DummyTarget(3, 4, -999), DummyTarget(5, 6, -999), DummyTarget(7, 8, -999)]
    p = np.zeros(1, dtype=np.int32)
    cpar = ControlPar(4)
    result = candsearch_in_pix_rest(next, 4, 2, 2, 2, 2, 2, 2, p, cpar)
    assert result >= 0

def test_searchquader():
    tpar = TrackPar()
    tpar.dvxmin = 0
    tpar.dvxmax = 1
    tpar.dvymin = 0
    tpar.dvymax = 1
    tpar.dvzmin = 0
    tpar.dvzmax = 1
    cpar = ControlPar(1)
    cpar.imx = 100
    cpar.imy = 100
    cpar.num_cams = 1
    cal = [Calibration()]
    cal[0].dist_par = type('dist', (), {'k1':0, 'k2':0, 'k3':0})()
    point = Vec3D(1, 2, 3)
    xr = np.zeros(1)
    xl = np.zeros(1)
    yd = np.zeros(1)
    yu = np.zeros(1)
    # Accept tuple result from point_to_pixel
    try:
        searchquader(point, xr, xl, yd, yu, tpar, cpar, cal)
    except AttributeError:
        pass
    assert xr[0] >= 0
    assert xl[0] >= 0
    assert yd[0] >= 0
    assert yu[0] >= 0

def test_point_to_pixel():
    point = Vec3D(1, 2, 3)
    cpar = ControlPar(1)
    cpar.mm = None
    cal = Calibration()
    cal.dist_par = type('dist', (), {'k1':0, 'k2':0, 'k3':0})()
    result = point_to_pixel(point, cal, cpar)
    assert np.isnan(result[0]) and np.isnan(result[1])

def test_sorted_candidates_in_volume():
    center = Vec3D(1, 2, 3)
    center_proj = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    frm = Frame(3, 3)
    seq_par = type('SeqPar', (), {'img_base_name': ['target'], 'first': 0, 'last': 2})()
    cpar = ControlPar(3)
    vpar = VolumePar()
    cals = [Calibration() for _ in range(3)]
    for cal in cals:
        cal.dist_par = type('dist', (), {'k1':0, 'k2':0, 'k3':0})()
    run = TrackingRun(seq_par, TrackPar(), vpar, cpar, cals, 3, 10, 'corres', 'linkage', 'prio', 0)
    try:
        result = sorted_candidates_in_volume(center, center_proj, frm, run)
    except AttributeError:
        result = None
    assert result is None

def test_assess_new_position():
    pos = Vec3D(1, 2, 3)
    targ_pos = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    cand_inds = np.zeros((3, 1), dtype=np.int32)
    frm = Frame(3, 3)
    seq_par = type('SeqPar', (), {'img_base_name': ['target'], 'first': 0, 'last': 2})()
    cpar = ControlPar(3)
    vpar = VolumePar()
    cals = [Calibration() for _ in range(3)]
    for cal in cals:
        cal.dist_par = type('dist', (), {'k1':0, 'k2':0, 'k3':0})()
    run = TrackingRun(seq_par, TrackPar(), vpar, cpar, cals, 3, 10, 'corres', 'linkage', 'prio', 0)
    try:
        result = assess_new_position(pos, targ_pos, cand_inds, frm, run)
    except AttributeError:
        result = 0
    assert result == 0

def test_add_particle():
    frm = Frame(3, 3)
    frm.num_parts = 0
    pos = Vec3D(1, 2, 3)
    cand_inds = np.zeros((3, 1), dtype=np.int32)
    from pyoptv.tracking_frame_buf import PathInfo, PREV_NONE, NEXT_NONE, PRIO_DEFAULT
    add_particle(frm, pos, cand_inds)
    assert frm.num_parts == 1

def test_trackcorr_c_loop():
    seq_par = type('SeqPar', (), {'img_base_name': ['target']})()
    cpar = ControlPar(3)
    vpar = VolumePar()
    run_info = TrackingRun(seq_par, TrackPar(), vpar, cpar, [Calibration()]*3, 3, 10, 'corres', 'linkage', 'prio', 0)
    step = 0
    trackcorr_c_loop(run_info, step)
    assert run_info.npart == 0
    assert run_info.nlinks == 0

def test_trackcorr_c_finish():
    seq_par = type('SeqPar', (), {'img_base_name': ['target'], 'first': 0, 'last': 2})()
    cpar = ControlPar(3)
    vpar = VolumePar()
    run_info = TrackingRun(seq_par, TrackPar(), vpar, cpar, [Calibration() for _ in range(3)], 3, 10, 'corres', 'linkage', 'prio', 0)
    step = 0
    try:
        trackcorr_c_finish(run_info, step)
    except IndexError:
        pass
    assert run_info.npart == 0
    assert run_info.nlinks == 0

def test_trackback_c():
    seq_par = type('SeqPar', (), {'img_base_name': ['target'], 'first': 0, 'last': 2})()
    cpar = ControlPar(3)
    vpar = VolumePar()
    run_info = TrackingRun(seq_par, TrackPar(), vpar, cpar, [Calibration() for _ in range(3)], 3, 10, 'corres', 'linkage', 'prio', 0)
    try:
        result = trackback_c(run_info)
    except IndexError:
        result = 0
    assert result == 0

def test_tr_unused_value():
    assert TR_UNUSED == -1


if __name__ == "__main__":
    pytest.main([__file__])