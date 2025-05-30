import pytest
import numpy as np
from pyoptv.imgcoord import flat_image_coord, img_coord, flat_to_dist
from pyoptv.calibration import Calibration
from pyoptv.parameters import MMNP


def test_flat_image_coord():
    pos = np.array([1.0, 2.0, 3.0])
    cal = Calibration()
    # Set up a simple camera with identity rotation and principal distance 3.0
    cal.ext_par.dm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cal.ext_par.x0 = 0.0
    cal.ext_par.y0 = 0.0
    cal.ext_par.z0 = 0.0
    cal.int_par.cc = 3.0
    # mm is now a valid MMNP instance (all air)
    mm = MMNP()
    x, y = flat_image_coord(pos, cal, mm)
    assert np.isclose(x, -1.0)
    assert np.isclose(y, -2.0)


def test_img_coord():
    pos = np.array([1.0, 2.0, 3.0])
    cal = Calibration()
    cal.ext_par.dm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cal.ext_par.x0 = 0.0
    cal.ext_par.y0 = 0.0
    cal.ext_par.z0 = 0.0
    cal.int_par.cc = 3.0
    # Set distortion parameters to zero
    cal.set_radial_distortion(np.array([0.0, 0.0, 0.0]))
    mm = MMNP()
    x, y = img_coord(pos, cal, mm)
    assert np.isclose(x, -1.0)
    assert np.isclose(y, -2.0)


def test_flat_to_dist():
    x, y = 1.0, 2.0
    cal = Calibration()
    cal.set_radial_distortion(np.array([0.0, 0.0, 0.0]))
    x_dist, y_dist = flat_to_dist(x, y, cal)
    assert np.isclose(x_dist, 1.0)
    assert np.isclose(y_dist, 2.0)


if __name__ == "__main__":
    pytest.main([__file__])