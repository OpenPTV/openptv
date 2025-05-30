import numpy as np
import pytest
from pyoptv.trafo import (
    old_metric_to_pixel,
    metric_to_pixel,
    old_pixel_to_metric,
    pixel_to_metric,
    distort_brown_affin,
    correct_brown_affin,
    correct_brown_affine_exact,
    flat_to_dist,
    dist_to_flat,
)
from pyoptv.calibration import Calibration, Interior, ap_52
from pyoptv.parameters import ControlPar

EPS = 1e-6

def test_old_metric_to_pixel():
    xc, yc = 0.0, 0.0
    imx, imy = 1024, 1008
    pix_x, pix_y = 0.010, 0.010
    field = 0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    assert abs(xp - 512.0) < EPS and abs(yp - 504.0) < EPS

    xc, yc = 1.0, 0.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    assert abs(xp - 612.0) < EPS and abs(yp - 504.0) < EPS

    xc, yc = 0.0, -1.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    assert abs(xp - 512.0) < EPS and abs(yp - 604.0) < EPS

def test_metric_to_pixel():
    xc, yc = 0.0, 0.0
    cpar = ControlPar()
    cpar.imx = 1024
    cpar.imy = 1008
    cpar.pix_x = 0.01
    cpar.pix_y = 0.01
    cpar.chfield = 0    
    xp, yp = metric_to_pixel(xc, yc, cpar)
    assert abs(xp - 512.0) < EPS and abs(yp - 504.0) < EPS

    xc, yc = 1.0, 0.0
    xp, yp = metric_to_pixel(xc, yc, cpar)
    assert abs(xp - 612.0) < EPS and abs(yp - 504.0) < EPS

    xc, yc = 0.0, -1.0
    xp, yp = metric_to_pixel(xc, yc, cpar)
    assert abs(xp - 512.0) < EPS and abs(yp - 604.0) < EPS

def test_old_pixel_to_metric():
    xc, yc = 0.0, 0.0
    imx, imy = 1024, 1008
    pix_x, pix_y = 0.010, 0.010
    field = 0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    xc1, yc1 = old_pixel_to_metric(xp, yp, imx, imy, pix_x, pix_y, field)
    assert abs(xc1 - xc) < EPS and abs(yc1 - yc) < EPS

    xc, yc = 1.0, 0.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    xc1, yc1 = old_pixel_to_metric(xp, yp, imx, imy, pix_x, pix_y, field)
    assert abs(xc1 - xc) < EPS and abs(yc1 - yc) < EPS

    xc, yc = 0.0, -1.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    xc1, yc1 = old_pixel_to_metric(xp, yp, imx, imy, pix_x, pix_y, field)
    assert abs(xc1 - xc) < EPS and abs(yc1 - yc) < EPS

def test_pixel_to_metric():
    xc, yc = 0.0, 0.0
    cpar = ControlPar()
    cpar.imx = 1024
    cpar.imy = 1008
    cpar.pix_x = 0.01
    cpar.pix_y = 0.01
    cpar.chfield = 0
    xp, yp = metric_to_pixel(xc, yc, cpar)
    xc1, yc1 = pixel_to_metric(xp, yp, cpar)
    assert abs(xc1 - xc) < EPS and abs(yc1 - yc) < EPS

    xc, yc = 1.0, 0.0
    xp, yp = metric_to_pixel(xc, yc, cpar)
    xc1, yc1 = pixel_to_metric(xp, yp, cpar)
    assert abs(xc1 - xc) < EPS and abs(yc1 - yc) < EPS

    xc, yc = 0.0, -1.0
    xp, yp = metric_to_pixel(xc, yc, cpar)
    xc1, yc1 = pixel_to_metric(xp, yp, cpar)
    assert abs(xc1 - xc) < EPS and abs(yc1 - yc) < EPS

def test_shear():
    x, y = 1.0, 1.0
    ap = ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    xp, yp = distort_brown_affin(x, y, ap)
    assert abs(xp - 0.158529) < EPS and abs(yp - 0.540302) < EPS

def test_shear_round_trip():
    x, y = -1.0, 10.0
    ap = ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1)
    xp, yp = distort_brown_affin(x, y, ap)
    x1, y1 = correct_brown_affin(xp, yp, ap)
    assert abs(x1 - x) < EPS and abs(y1 - y) < EPS

    x, y = 0.5, -5.0
    xp, yp = distort_brown_affin(x, y, ap)
    x1, y1 = correct_brown_affin(xp, yp, ap)
    assert abs(x1 - x) < EPS and abs(y1 - y) < EPS

def test_dummy_distortion_round_trip():
    x, y = 1.0, 1.0
    ap = ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    xres, yres = distort_brown_affin(x, y, ap)
    xres, yres = correct_brown_affin(xres, yres, ap)
    assert abs(xres - x) < EPS and abs(yres - y) < EPS

def test_radial_distortion_round_trip():
    x, y = 1.0, 1.0
    ap = ap_52(0.05, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    iter_eps = 0.05
    xres, yres = distort_brown_affin(x, y, ap)
    xres, yres = correct_brown_affin(xres, yres, ap)
    assert abs(xres - x) < iter_eps and abs(yres - y) < iter_eps

def test_dist_flat_round_trip():
    x, y = 10.0, 10.0
    iter_eps = 1e-3
    cal = Calibration(int_par=Interior(1.5, 1.5, 60.), 
                      added_par=ap_52(0.0005, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                      )
    xres, yres = flat_to_dist(x, y, cal)
    xres, yres = dist_to_flat(xres, yres, cal, iter_eps)
    assert abs(xres - x) < iter_eps and abs(yres - y) < iter_eps

if __name__ == "__main__":
    pytest.main([__file__])