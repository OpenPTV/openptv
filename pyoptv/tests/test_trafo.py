import pytest
import numpy as np
from pyoptv.trafo import (
    old_pixel_to_metric,
    pixel_to_metric,
    old_metric_to_pixel,
    metric_to_pixel,
    distort_brown_affin,
    correct_brown_affin,
    correct_brown_affine_exact,
    flat_to_dist,
    dist_to_flat,
    ControlPar,
    Calibration,
    AddedPar,
)

def test_old_pixel_to_metric():
    x_pixel, y_pixel = 100, 200
    im_size_x, im_size_y = 640, 480
    pix_size_x, pix_size_y = 0.01, 0.01
    y_remap_mode = 0
    x_metric, y_metric = old_pixel_to_metric(x_pixel, y_pixel, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode)
    assert np.isclose(x_metric, -2.7)
    assert np.isclose(y_metric, 1.4)

def test_pixel_to_metric():
    x_pixel, y_pixel = 100, 200
    parameters = ControlPar(640, 480, 0.01, 0.01, 0)
    x_metric, y_metric = pixel_to_metric(x_pixel, y_pixel, parameters)
    assert np.isclose(x_metric, -2.7)
    assert np.isclose(y_metric, 1.4)

def test_old_metric_to_pixel():
    x_metric, y_metric = -2.7, 1.4
    im_size_x, im_size_y = 640, 480
    pix_size_x, pix_size_y = 0.01, 0.01
    y_remap_mode = 0
    x_pixel, y_pixel = old_metric_to_pixel(x_metric, y_metric, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode)
    assert np.isclose(x_pixel, 100)
    assert np.isclose(y_pixel, 200)

def test_metric_to_pixel():
    x_metric, y_metric = -2.7, 1.4
    parameters = ControlPar(640, 480, 0.01, 0.01, 0)
    x_pixel, y_pixel = metric_to_pixel(x_metric, y_metric, parameters)
    assert np.isclose(x_pixel, 100)
    assert np.isclose(y_pixel, 200)

def test_distort_brown_affin():
    x, y = 1.0, 2.0
    ap = AddedPar(0.1, 0.01, 0.001, 0.01, 0.01, 1.0, 0.1)
    x1, y1 = distort_brown_affin(x, y, ap)
    assert np.isclose(x1, 1.123)
    assert np.isclose(y1, 2.246)

def test_correct_brown_affin():
    x, y = 1.0, 2.0
    ap = AddedPar(0.1, 0.01, 0.001, 0.01, 0.01, 1.0, 0.1)
    x1, y1 = correct_brown_affin(x, y, ap)
    assert np.isclose(x1, 0.987)
    assert np.isclose(y1, 1.975)

def test_correct_brown_affine_exact():
    x, y = 1.0, 2.0
    ap = AddedPar(0.1, 0.01, 0.001, 0.01, 0.01, 1.0, 0.1)
    tol = 1e-8
    x1, y1 = correct_brown_affine_exact(x, y, ap, tol)
    assert np.isclose(x1, 0.987)
    assert np.isclose(y1, 1.975)

def test_flat_to_dist():
    flat_x, flat_y = 1.0, 2.0
    cal = Calibration(0.1, 0.1, AddedPar(0.1, 0.01, 0.001, 0.01, 0.01, 1.0, 0.1))
    dist_x, dist_y = flat_to_dist(flat_x, flat_y, cal)
    assert np.isclose(dist_x, 1.123)
    assert np.isclose(dist_y, 2.246)

def test_dist_to_flat():
    dist_x, dist_y = 1.0, 2.0
    cal = Calibration(0.1, 0.1, AddedPar(0.1, 0.01, 0.001, 0.01, 0.01, 1.0, 0.1))
    tol = 1e-8
    flat_x, flat_y = dist_to_flat(dist_x, dist_y, cal, tol)
    assert np.isclose(flat_x, 0.9)
    assert np.isclose(flat_y, 1.9)
