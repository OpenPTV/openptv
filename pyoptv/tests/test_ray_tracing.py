import pytest
import numpy as np
from pyoptv.ray_tracing import ray_tracing

def test_ray_tracing():
    x = 100.0
    y = 100.0
    cal = create_calibration()
    mm = create_multimed()
    X = np.zeros(3)
    a = np.zeros(3)
    
    ray_tracing(x, y, cal, mm, X, a)
    
    # Expected values from C test
    expected_X = np.array([110.406944, 88.325788, 0.988076])
    expected_a = np.array([0.387960, 0.310405, -0.867834])
    
    assert np.allclose(X, expected_X, atol=1e-5)
    assert np.allclose(a, expected_a, atol=1e-5)

def create_calibration():
    class Calibration:
        def __init__(self):
            self.int_par = self.IntPar()
            self.ext_par = self.ExtPar()
            self.glass_par = self.GlassPar()
        
        class IntPar:
            def __init__(self):
                self.cc = 100.0  # Camera constant from C test
        
        class ExtPar:
            def __init__(self):
                # Rotation matrix from C test
                self.dm = np.array([[1.0, 0.2, -0.3], 
                                   [0.2, 1.0, 0.0],
                                   [-0.3, 0.0, 1.0]])
                self.x0 = 0.0
                self.y0 = 0.0
                self.z0 = 100.0
        
        class GlassPar:
            def __init__(self):
                self.vec_x = 0.0001
                self.vec_y = 0.00001
                self.vec_z = 1.0
    
    return Calibration()

def create_multimed():
    class Multimed:
        def __init__(self):
            self.d = [5.0, 0.0, 0.0]  # From C test
            self.n1 = 1.0
            self.n2 = [1.49, 0.0, 0.0]  # From C test
            self.n3 = 1.33
    
    return Multimed()

if __name__ == "__main__":
    pytest.main()
