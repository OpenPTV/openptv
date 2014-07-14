"""
Check that the bindings do what they are expected to.
The test expects to be run from the py_bind/test/ directory for now,
using the nose test harness [1].

References:
[1] https://nose.readthedocs.org/en/latest/
"""

import unittest
import optv.ray_tracing as rt
import numpy as np

class TestRayTracing(unittest.TestCase):

    @staticmethod
    def get_dummy_mm_np():
        ret = rt.pmm_np()
        ret.nlay = 3
        ret.n1 = 1.
        ret.n2 = np.array([1.49,0.,0])
        ret.d = np.array([5.,0.,0])
        ret.n3 = 1.33
        ret.lut = 1

        return ret

    @staticmethod
    def get_dummy_Interior():
        return {'xh':0. , 'yh':0. , 'cc':100.}


    @staticmethod
    def get_dummy_Glass():
        return {'vec_x': 0.0001,'vec_y': 0.00001,'vec_z': 1.}

    @staticmethod
    def get_dummy_Exterior():
        ret = rt.pExterior()
        ret.z0 = 100
        ret.dm = np.array([[1.0, 0.2, -0.3], 
            [0.2, 1.0, 0.0],
            [-0.3, 0.0, 1.0]])

        return ret 
    
    def test_ray_tracing(self):
        """Testing ray tracing against data from ray_tracing.c check testing."""
        
        mm = self.get_dummy_mm_np()

        Ex = self.get_dummy_Exterior()

        I = self.get_dummy_Interior()

        G = self.get_dummy_Glass()

        
        tracer = rt.Ray_tracing()
        tracer.force_set_exterior(Ex)
        tracer.force_set_interior(I)
        tracer.force_set_glass(G)
        tracer.force_set_mm_np(mm)

        input_X = (100.,100.00)
        output_X, output_A = tracer.trace(input_X)

        self.failUnlessAlmostEqual(
            np.max(
                np.abs(
                    np.array(output_X)-np.array((110.406944, 88.325788, 0.988076))
                )
            ),0. , places = 5)
        self.failUnlessAlmostEqual(
            np.max(
                np.abs(            
                    np.array(output_A)-np.array((0.387960,0.310405,-0.867834))
                )
            ),0. , places = 5)

