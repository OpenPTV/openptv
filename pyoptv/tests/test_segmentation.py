import pytest
import numpy as np
from pyoptv.segmentation import targ_rec, peak_fit

def test_targ_rec():
    img = np.array([[0, 0, 0, 0, 0],
                    [0, 10, 10, 10, 0],
                    [0, 10, 20, 10, 0],
                    [0, 10, 10, 10, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8)
    targ_par = {
        'gvthres': [5],
        'discont': 5,
        'nnmin': 1,
        'nnmax': 100,
        'nxmin': 1,
        'nxmax': 100,
        'nymin': 1,
        'nymax': 100,
        'sumg_min': 10
    }
    cpar = {'imx': 5, 'imy': 5}
    xmin, xmax, ymin, ymax = 0, 5, 0, 5
    num_cam = 0
    targets = targ_rec(img, targ_par, xmin, xmax, ymin, ymax, cpar, num_cam)
    assert len(targets) == 1
    assert targets[0]['n'] == 9
    assert targets[0]['sumg'] == 85
    assert np.isclose(targets[0]['x'], 2.0)
    assert np.isclose(targets[0]['y'], 2.0)

def test_peak_fit():
    img = np.array([[0, 0, 0, 0, 0],
                    [0, 10, 10, 10, 0],
                    [0, 10, 20, 10, 0],
                    [0, 10, 10, 10, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8)
    targ_par = {
        'gvthres': [5],
        'discont': 5,
        'nnmin': 1,
        'nnmax': 100,
        'nxmin': 1,
        'nxmax': 100,
        'nymin': 1,
        'nymax': 100,
        'sumg_min': 10
    }
    cpar = {'imx': 5, 'imy': 5}
    xmin, xmax, ymin, ymax = 0, 5, 0, 5
    num_cam = 0
    targets = peak_fit(img, targ_par, xmin, xmax, ymin, ymax, cpar, num_cam)
    assert len(targets) == 1
    assert targets[0]['n'] == 9
    assert targets[0]['sumg'] == 85
    assert np.isclose(targets[0]['x'], 2.0)
    assert np.isclose(targets[0]['y'], 2.0)
