from pyoptv.calibration import Calibration
import pytest
import numpy as np
from pyoptv.ray_tracing import ray_tracing

def test_ray_tracing():
    x = 100.0
    y = 100.0
    cal = create_calibration()
    mm = create_multimed()
    X, a = ray_tracing(x, y, cal, mm)

    # Expected values from C test
    expected_X = np.array([110.406944, 88.325788, 0.988076])
    expected_a = np.array([0.387960, 0.310405, -0.867834])

    assert np.allclose([X.x, X.y, X.z], expected_X, atol=1e-5)
    assert np.allclose([a.x, a.y, a.z], expected_a, atol=1e-5)

def create_calibration():
    cal = Calibration()
    cal.int_par.cc = 100.0  # Camera constant from C test
    cal.ext_par.dm = np.array([[1.0, 0.2, -0.3], 
                                [0.2, 1.0, 0.0],
                                [-0.3, 0.0, 1.0]])
    cal.ext_par.x0 = 0.0
    cal.ext_par.y0 = 0.0
    cal.ext_par.z0 = 100.0
    cal.glass_par.vec_x = 0.0001
    cal.glass_par.vec_y = 0.00001
    cal.glass_par.vec_z = 1.0

    
    return cal

def create_multimed():
    from pyoptv.parameters import MMNP
    # Create a mock MMNP object with the same structure as in the C test
    mm = MMNP()
    mm.d = [5.0, 0.0, 0.0]  # Thickness of layers
    mm.n1 = 1.0  # Refractive index of the first medium
    mm.n2 = [1.49, 0.0, 0.0]  # Refractive index of the second medium
    mm.n3 = 1.33  # Refractive index of the third medium

    return mm

if __name__ == "__main__":
    pytest.main([__file__])
