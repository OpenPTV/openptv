import pytest
import numpy as np
from pyoptv.imgcoord import flat_image_coord, img_coord, flat_to_dist, flat_image_coord_numba, img_coord_numba, flat_to_dist_numba, plot_image_coords

def test_flat_image_coord():
    pos = np.array([1.0, 2.0, 3.0])
    cal = {
        'ext_par': {
            'dm': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'x0': 0.0,
            'y0': 0.0,
            'z0': 0.0
        },
        'int_par': {
            'cc': 1.0
        }
    }
    mm = None
    x, y = flat_image_coord(pos, cal, mm)
    assert np.isclose(x, -1.0)
    assert np.isclose(y, -2.0)

def test_img_coord():
    pos = np.array([1.0, 2.0, 3.0])
    cal = {
        'ext_par': {
            'dm': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'x0': 0.0,
            'y0': 0.0,
            'z0': 0.0
        },
        'int_par': {
            'cc': 1.0
        },
        'dist_par': {
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0
        }
    }
    mm = None
    x, y = img_coord(pos, cal, mm)
    assert np.isclose(x, -1.0)
    assert np.isclose(y, -2.0)

def test_flat_to_dist():
    x, y = 1.0, 2.0
    cal = {
        'dist_par': {
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0
        }
    }
    x_dist, y_dist = flat_to_dist(x, y, cal)
    assert np.isclose(x_dist, 1.0)
    assert np.isclose(y_dist, 2.0)

def test_flat_image_coord_numba():
    pos = np.array([1.0, 2.0, 3.0])
    cal = {
        'ext_par': {
            'dm': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'x0': 0.0,
            'y0': 0.0,
            'z0': 0.0
        },
        'int_par': {
            'cc': 1.0
        }
    }
    mm = None
    x, y = flat_image_coord_numba(pos, cal, mm)
    assert np.isclose(x, -1.0)
    assert np.isclose(y, -2.0)

def test_img_coord_numba():
    pos = np.array([1.0, 2.0, 3.0])
    cal = {
        'ext_par': {
            'dm': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'x0': 0.0,
            'y0': 0.0,
            'z0': 0.0
        },
        'int_par': {
            'cc': 1.0
        },
        'dist_par': {
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0
        }
    }
    mm = None
    x, y = img_coord_numba(pos, cal, mm)
    assert np.isclose(x, -1.0)
    assert np.isclose(y, -2.0)

def test_flat_to_dist_numba():
    x, y = 1.0, 2.0
    cal = {
        'dist_par': {
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0
        }
    }
    x_dist, y_dist = flat_to_dist_numba(x, y, cal)
    assert np.isclose(x_dist, 1.0)
    assert np.isclose(y_dist, 2.0)

def test_plot_image_coords():
    pos = np.array([1.0, 2.0, 3.0])
    cal = {
        'ext_par': {
            'dm': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'x0': 0.0,
            'y0': 0.0,
            'z0': 0.0
        },
        'int_par': {
            'cc': 1.0
        },
        'dist_par': {
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0
        }
    }
    mm = None
    plot_image_coords(pos, cal, mm)
