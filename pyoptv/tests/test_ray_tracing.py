import numpy as np
import pytest
from pyoptv.calibration import Calibration, Exterior, Interior, Glass, ap_52
from pyoptv.ray_tracing import ray_tracing
from pyoptv.parameters import MMNP as mm_np

EPS = 1e-6

def test_ray_tracing():
    # input
    x = 100.0
    y = 100.0

    test_Ex = Exterior(
        x0=0.0, y0=0.0, z0=100.0,
        omega=0.0, phi=0.0, kappa=0.0,
        dm=np.array([[1.0, 0.2, -0.3],
                     [0.2, 1.0, 0.0],
                     [-0.3, 0.0, 1.0]])
    )
    test_I = Interior(xh=0.0, yh=0.0, cc=100.0)
    test_G = Glass(0.0001, 0.00001, 1.0)
    test_addp = ap_52(0., 0., 0., 0., 0., 1., 0.)
    test_cal = Calibration(test_Ex, test_I, test_G, test_addp)


    test_mm = mm_np(
    	nlay=3, 
    	n1=1.0, 
    	n2= [1.49, 0.0, 0.0], 
    	d = [5.0, 0.0, 0.0],
    	n3 = 1.33
    )

    X, a = ray_tracing(x, y, test_cal, test_mm)

    assert np.allclose(X, [110.406944, 88.325788, 0.988076], atol=EPS), \
        f"Expected [110.406944, 88.325788, 0.988076] but found {X}"

    assert np.allclose(a, [0.387960, 0.310405, -0.867834], atol=EPS), \
        f"Expected [0.387960, 0.310405, -0.867834] but found {a}"

if __name__ == "__main__":
    pytest.main([__file__])
