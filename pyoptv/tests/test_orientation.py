import pytest
import numpy as np
from pyoptv.orientation import (
    skew_midpoint, point_position, weighted_dumbbell_precision, num_deriv_exterior,
    orient, raw_orient, read_man_ori_fix, read_orient_par, OrientPar
)
from pyoptv.ray_tracing import ray_tracing

def test_skew_midpoint():
    vert1 = np.array([0, 0, 0])
    direct1 = np.array([1, 0, 0])
    vert2 = np.array([0, 1, 0])
    direct2 = np.array([0, 1, 0])
    dist, res = skew_midpoint(vert1, direct1, vert2, direct2)
    assert np.isclose(dist, 1.0)
    np.testing.assert_array_almost_equal(res, np.array([0.5, 1.0, 0.0]))

def test_point_position():
    targets = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    num_cams = 4
    multimed_pars = None
    cals = [None] * num_cams
    dist, res = point_position(targets, num_cams, multimed_pars, cals)
    assert np.isclose(dist, 0.0)
    np.testing.assert_array_almost_equal(res, np.array([1.5, 1.5, 0.0]))

def test_weighted_dumbbell_precision():
    targets = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    num_targs = 4
    num_cams = 4
    multimed_pars = None
    cals = [None] * num_cams
    db_length = 1.0
    db_weight = 1.0
    precision = weighted_dumbbell_precision(targets, num_targs, num_cams, multimed_pars, cals, db_length, db_weight)
    assert np.isclose(precision, 0.0)

def test_num_deriv_exterior():
    cal = None
    cpar = None
    dpos = 0.0001
    dang = 0.0001
    pos = np.array([0, 0, 0])
    x_ders, y_ders = num_deriv_exterior(cal, cpar, dpos, dang, pos)
    np.testing.assert_array_almost_equal(x_ders, np.zeros(6))
    np.testing.assert_array_almost_equal(y_ders, np.zeros(6))

def test_orient():
    cal_in = None
    cpar = None
    nfix = 4
    fix = [None] * nfix
    pix = [None] * nfix
    flags = OrientPar()
    sigmabeta = np.zeros(20)
    resi = orient(cal_in, cpar, nfix, fix, pix, flags, sigmabeta)
    assert resi is None

def test_raw_orient():
    cal = None
    cpar = None
    nfix = 4
    fix = [None] * nfix
    pix = [None] * nfix
    stopflag = raw_orient(cal, cpar, nfix, fix, pix)
    assert not stopflag

def test_read_man_ori_fix():
    fix4 = [None] * 4
    calblock_filename = "calblock.txt"
    man_ori_filename = "man_ori.txt"
    cam = 1
    num_match = read_man_ori_fix(fix4, calblock_filename, man_ori_filename, cam)
    assert num_match == 0

def test_read_orient_par():
    filename = "orient_par.txt"
    params = read_orient_par(filename)
    assert isinstance(params, OrientPar)
    assert params.useflag == 0
    assert params.ccflag == 0
    assert params.xhflag == 0
    assert params.yhflag == 0
    assert params.k1flag == 0
    assert params.k2flag == 0
    assert params.k3flag == 0
    assert params.p1flag == 0
    assert params.p2flag == 0
    assert params.scxflag == 0
    assert params.sheflag == 0
    assert params.interfflag == 0

if __name__ == "__main__":
    pytest.main([__file__])