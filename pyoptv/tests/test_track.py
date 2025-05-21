import pytest
import numpy as np
from pyoptv.track import (
    TrackingRun, FrameBuffer, PathInfo, FoundPix, Vec3D, Vec2D,
    vec_scalar_mul, vec_subt, vec_diff_norm, vec_dot, vec_norm, vec_set, vec_copy,
    reset_foundpix_array, copy_foundpix_array, register_closest_neighbs, search_volume_center_moving,
    predict, pos3d_in_bounds, angle_acc, candsearch_in_pix, candsearch_in_pix_rest, searchquader,
    sort_candidates_by_freq, sort, point_to_pixel, sorted_candidates_in_volume, assess_new_position,
    add_particle, trackcorr_c_loop, trackcorr_c_finish, trackback_c
)

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
        assert item.ftnr == -1
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
    targets = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    reg = [FoundPix(-1, 0, [0, 0, 0]) for _ in range(4)]
    register_closest_neighbs(targets, 3, 0, 2, 2, 2, 2, reg, None)
    assert reg[0].ftnr == 0
    assert reg[1].ftnr == 1
    assert reg[2].ftnr == 2
    assert reg[3].ftnr == -1

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
    pos = Vec3D(1, 2, 3)
    bounds = Vec3D(0, 0, 0)
    assert pos3d_in_bounds(pos, bounds) == False
    bounds = Vec3D(2, 3, 4)
    assert pos3d_in_bounds(pos, bounds) == True

def test_angle_acc():
    start = Vec3D(1, 2, 3)
    pred = Vec3D(4, 5, 6)
    cand = Vec3D(7, 8, 9)
    angle, acc = angle_acc(start, pred, cand)
    assert np.isclose(angle, 0)
    assert np.isclose(acc, 5.196152422706632)

def test_candsearch_in_pix():
    next = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    p = np.zeros(4, dtype=np.int32)
    result = candsearch_in_pix(next, 3, 2, 2, 2, 2, p, None)
    assert result == 3
    assert p[0] == 0
    assert p[1] == 1
    assert p[2] == 2
    assert p[3] == -999

def test_candsearch_in_pix_rest():
    next = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    p = np.zeros(1, dtype=np.int32)
    result = candsearch_in_pix_rest(next, 3, 2, 2, 2, 2, p, None)
    assert result == 1
    assert p[0] == 0

def test_searchquader():
    point = Vec3D(1, 2, 3)
    xr = np.zeros(3)
    xl = np.zeros(3)
    yd = np.zeros(3)
    yu = np.zeros(3)
    searchquader(point, xr, xl, yd, yu, None, None, None)
    assert xr[0] == 0
    assert xl[0] == 0
    assert yd[0] == 0
    assert yu[0] == 0

def test_sort_candidates_by_freq():
    item = [FoundPix(1, 2, [1, 1, 1]), FoundPix(2, 3, [1, 1, 1]), FoundPix(3, 4, [1, 1, 1])]
    result = sort_candidates_by_freq(item, 3)
    assert result == 3
    assert item[0].ftnr == 3
    assert item[1].ftnr == 2
    assert item[2].ftnr == 1

def test_sort():
    a = np.array([3, 1, 2])
    b = np.array([1, 2, 3])
    sort(3, a, b)
    assert np.array_equal(a, np.array([1, 2, 3]))
    assert np.array_equal(b, np.array([2, 3, 1]))

def test_point_to_pixel():
    point = Vec3D(1, 2, 3)
    result = point_to_pixel(point, None, None)
    assert result.x == 0
    assert result.y == 0

def test_sorted_candidates_in_volume():
    center = Vec3D(1, 2, 3)
    center_proj = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    frm = FrameBuffer(3, 3, None, None, None)
    run = TrackingRun(None, None, None, None, None, None, 0, 0, 0, 0, 0)
    result = sorted_candidates_in_volume(center, center_proj, frm, run)
    assert result is None

def test_assess_new_position():
    pos = Vec3D(1, 2, 3)
    targ_pos = [Vec2D(1, 2), Vec2D(3, 4), Vec2D(5, 6)]
    cand_inds = np.zeros((3, 1), dtype=np.int32)
    frm = FrameBuffer(3, 3, None, None, None)
    run = TrackingRun(None, None, None, None, None, None, 0, 0, 0, 0, 0)
    result = assess_new_position(pos, targ_pos, cand_inds, frm, run)
    assert result == 0

def test_add_particle():
    frm = FrameBuffer(3, 3, None, None, None)
    pos = Vec3D(1, 2, 3)
    cand_inds = np.zeros((3, 1), dtype=np.int32)
    add_particle(frm, pos, cand_inds)
    assert frm.num_parts == 1

def test_trackcorr_c_loop():
    run_info = TrackingRun(None, None, None, None, None, None, 0, 0, 0, 0, 0)
    step = 0
    trackcorr_c_loop(run_info, step)
    assert run_info.npart == 0
    assert run_info.nlinks == 0

def test_trackcorr_c_finish():
    run_info = TrackingRun(None, None, None, None, None, None, 0, 0, 0, 0, 0)
    step = 0
    trackcorr_c_finish(run_info, step)
    assert run_info.npart == 0
    assert run_info.nlinks == 0

def test_trackback_c():
    run_info = TrackingRun(None, None, None, None, None, None, 0, 0, 0, 0, 0)
    result = trackback_c(run_info)
    assert result == 0
