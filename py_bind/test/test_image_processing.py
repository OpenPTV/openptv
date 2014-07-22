"""
Check that the bindings do what they are expected to.
The test expects to be run from the py_bind/test/ directory for now,
using the nose test harness [1].

References:
[1] https://nose.readthedocs.org/en/latest/
"""

import unittest
import optv.image_processing as ip
import numpy as np

class TestImageProcessing(unittest.TestCase):
    
    def test_lowpass_3(self):
        """Testing lowpass_3  against synthetic data and Lena."""
        
        a = 128*np.ones((3,3),dtype=np.uint8)
        b = np.copy(a)
        
        ip.lowpass_3(a,b)

        self.failUnlessAlmostEqual(
            np.max(
                np.abs(
                    b.flatten() - np.array((128,136,141,154,167,154,139,138,123))
                )
            ), 0. , places = 5)

