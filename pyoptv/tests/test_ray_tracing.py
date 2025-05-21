import pytest
import numpy as np
from pyoptv.ray_tracing import ray_tracing

def test_ray_tracing():
    x = 1.0
    y = 2.0
    cal = create_calibration()
    mm = create_multimed()
    X = np.zeros(3)
    a = np.zeros(3)
    
    ray_tracing(x, y, cal, mm, X, a)
    
    assert np.allclose(X, expected_X(), atol=1e-6)
    assert np.allclose(a, expected_a(), atol=1e-6)

def create_calibration():
    class Calibration:
        def __init__(self):
            self.int_par = self.IntPar()
            self.ext_par = self.ExtPar()
            self.glass_par = self.GlassPar()
        
        class IntPar:
            def __init__(self):
                self.cc = 1.0
        
        class ExtPar:
            def __init__(self):
                self.dm = np.eye(3)
                self.x0 = 0.0
                self.y0 = 0.0
                self.z0 = 0.0
        
        class GlassPar:
            def __init__(self):
                self.vec_x = 1.0
                self.vec_y = 0.0
                self.vec_z = 0.0
    
    return Calibration()

def create_multimed():
    class Multimed:
        def __init__(self):
            self.d = [1.0]
            self.n1 = 1.0
            self.n2 = [1.0]
            self.n3 = 1.0
    
    return Multimed()

def expected_X():
    return np.array([1.0, 2.0, 3.0])

def expected_a():
    return np.array([0.0, 1.0, 0.0])

if __name__ == "__main__":
    pytest.main()
