"""
Check that the bindings do what they are expected to.
The test expects to be run from the py_bind/test/ directory for now,
using the nose test harness [1].

References:
[1] https://nose.readthedocs.org/en/latest/
"""

import unittest
import numpy as np
from optv.intersect import py_intersect

class TestTargets(unittest.TestCase):
    def test_interesect(self):
        """Testing intersect subroutine"""
        pos1 = np.array([1.0,0.0,0.0])
        vec1 = np.array([0.0,-0.707,1.0])
        pos2 = np.array([0.0,1.0,0.0])
        vec2 = np.array([0.0,0.707,1.0])
        x,y,z = py_intersect(pos1,vec1,pos2,vec2)


        self.failUnlessEqual(x, 0.5)
        self.failUnlessEqual(y, 0.5)
        self.failUnlessEqual(round(z,3), -0.707)
        
        pos1 = np.array([0,0,0],dtype=np.float)
        vec1 = np.array([0,0,1],dtype=np.float)
        pos2 = np.array([0,0,0],dtype=np.float)
        vec2 = np.array([0,0,1],dtype=np.float)
        x,y,z = py_intersect(pos1,vec1,pos2,vec2)


        self.failUnlessEqual(x, 1000000.)
        self.failUnlessEqual(y, 1000000.)
        self.failUnlessEqual(z, 1000000.)
