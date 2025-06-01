import pytest
import numpy as np
from pyoptv.epi import epipolar_curve, find_candidate
from pyoptv.parameters import ControlPar, VolumePar
from pyoptv.calibration import Calibration
from pyoptv.trafo import metric_to_pixel
from dataclasses import dataclass

@dataclass
class Coord2D:
    pnr: int
    x: float
    y: float

@pytest.fixture
def basic_calibration():
    cal1 = Calibration()
    cal2 = Calibration()

    # Camera 1 at (100, 0, 1000), looking towards origin, rotated -atan(100/1000) around Y
    cal1.ext_par.x0, cal1.ext_par.y0, cal1.ext_par.z0 = 100.0, 0.0, 1000.0
    cal1.ext_par.omega = 0.0  # rotation around X
    cal1.ext_par.phi = -np.arctan2(100.0, 1000.0)  # rotation around Y towards origin
    cal1.ext_par.kappa = 0.0  # rotation around Z

    # Camera 2 at (-100, 0, 1000), looking towards origin, rotated +atan(100/1000) around Y
    cal2.ext_par.x0, cal2.ext_par.y0, cal2.ext_par.z0 = -100.0, 0.0, 1000.0
    cal2.ext_par.omega = 0.0
    cal2.ext_par.phi = np.arctan2(100.0, 1000.0)  # rotation around Y towards origin
    cal2.ext_par.kappa = 0.0

    cal1.int_par.cc = 1.0
    cal2.int_par.cc = 1.0

    # Set a valid glass vector for both cameras
    cal1.set_glass_vec(np.array([0.0, 0.0, 1.0]))
    cal2.set_glass_vec(np.array([0.0, 0.0, 1.0]))

    return cal1, cal2

@pytest.fixture
def basic_parameters():
    cpar = ControlPar(num_cams=2)
    cpar.set_image_size((256, 256))
    cpar.set_pixel_size((0.01, 0.01))

    vpar = VolumePar()
    vpar.X_lay = [-10.0, 10.0]
    vpar.Zmin_lay = [-10, -10]
    vpar.Zmax_lay = [10, 10]

    return cpar, vpar

def test_epipolar_curve_basic(basic_calibration, basic_parameters):
    cal1, cal2 = basic_calibration
    cpar, vpar = basic_parameters

    # Known point in camera 1
    point_cam1 = (0.0, 0.0)

    # Compute epipolar curve in camera 2
    curve = epipolar_curve(point_cam1, cal1, cal2, 5, cpar, vpar)

    # Analytical solution for the epipolar curve
    expected_curve = np.array([
        [0.0, -0.1],
        [0.0, -0.05],
        [0.0, 0.0],
        [0.0, 0.05],
        [0.0, 0.1],
    ])

    assert np.allclose(curve, expected_curve, atol=1e-6), "Epipolar curve does not match expected values"

def test_find_candidate_basic(basic_calibration, basic_parameters):
    cal1, cal2 = basic_calibration
    cpar, vpar = basic_parameters

    # Known point in camera 1
    point_cam1 = (0.0, 0.0)

    # Compute epipolar curve in camera 2
    curve = epipolar_curve(point_cam1, cal1, cal2, 5, cpar, vpar)

    # Known candidates in camera 2
    candidates = [
        (0.0, -0.1),
        (0.0, -0.05),
        (0.0, 0.0),
        (0.0, 0.05),
        (0.0, 0.1),
    ]

    # Convert candidates to Coord2D objects
    metric_candidates = [Coord2D(pnr=i, x=x, y=y) for i, (x, y) in enumerate(candidates)]

    # Run find_candidate
    count, found_candidates = find_candidate(metric_candidates, [], len(metric_candidates),
                                             curve[0][0], curve[0][1], curve[-1][0], curve[-1][1],
                                             1, 1, 1, 1, vpar, cpar, cal2)

    assert count == len(candidates), "Number of candidates found does not match expected"
    assert np.allclose(found_candidates, candidates, atol=1e-6), "Candidates do not match expected values"
