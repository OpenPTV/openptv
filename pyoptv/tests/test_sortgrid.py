import pytest
import numpy as np
from pyoptv.sortgrid import img_coord, metric_to_pixel, sortgrid, nearest_neighbour_pix, read_sortgrid_par, read_calblock

def test_img_coord():
    fix = np.array([0.0, 0.0, 0.0])
    cal = None
    mm = None
    xp, yp = img_coord(fix, cal, mm)
    assert xp == 0.0
    assert yp == 0.0

def test_metric_to_pixel():
    xp, yp = 0.0, 0.0
    cpar = None
    calib_point = metric_to_pixel(xp, yp, cpar)
    assert np.allclose(calib_point, np.array([0.0, 0.0]))

def test_sortgrid():
    cal = None
    cpar = {'mm': None, 'imx': 1000, 'imy': 1000}
    nfix = 1
    fix = np.array([{'x': 0.0, 'y': 0.0, 'z': 0.0}])
    num = 1
    eps = 1.0
    pix = np.array([{'x': 0.0, 'y': 0.0}])
    sorted_pix = sortgrid(cal, cpar, nfix, fix, num, eps, pix)
    assert sorted_pix[0]['pnr'] == 0

def test_nearest_neighbour_pix():
    pix = np.array([{'x': 0.0, 'y': 0.0}])
    num = 1
    x, y = 0.0, 0.0
    eps = 1.0
    pnr = nearest_neighbour_pix(pix, num, x, y, eps)
    assert pnr == 0

def test_read_sortgrid_par(tmp_path):
    filename = tmp_path / "sortgrid.par"
    with open(filename, 'w') as f:
        f.write("1\n")
    eps = read_sortgrid_par(filename)
    assert eps == 1

def test_read_calblock(tmp_path):
    filename = tmp_path / "calblock.txt"
    with open(filename, 'w') as f:
        f.write("1 0.0 0.0 0.0\n")
    fix, num_points = read_calblock(filename)
    assert num_points == 1
    assert fix[0]['x'] == 0.0
    assert fix[0]['y'] == 0.0
    assert fix[0]['z'] == 0.0
