"""
Tests for the correspondences bindings, including supporting infrastructure
such as the MatchedCoordinates structure.
"""

import unittest
import numpy as np

from optv.parameters import ControlParams
from optv.calibration import Calibration
from optv.tracking_framebuf import read_targets
from optv.correspondences import MatchedCoords

class TestMatchedCoords(unittest.TestCase):
    def test_instantiate(self):
        """Creating a MatchedCoords object"""
        cal = Calibration()
        cpar = ControlParams(4)
        
        cal.from_file(
            "testing_fodder/calibration/cam1.tif.ori",
            "testing_fodder/calibration/cam2.tif.addpar")
        cpar.read_control_par("testing_fodder/corresp/control.par")
        targs = read_targets("testing_fodder/frame/cam1.", 333)
        
        mc = MatchedCoords(targs, cpar, cal)
        pos, pnr = mc.as_arrays()
        
        # x sorted?
        self.failUnless(np.all(pos[1:,0] > pos[:-1,0]))
        
        # Manually verified order for the loaded data:
        np.testing.assert_array_equal(
            pnr, np.r_[6, 11, 10,  8,  1,  4,  7,  0,  2,  9,  5,  3, 12])
        
