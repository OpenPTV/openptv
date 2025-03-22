"""
Tests for the correspondences bindings, including supporting infrastructure
such as the MatchedCoordinates structure.
"""

import unittest
import numpy as np

from optv.parameters import ControlParams, VolumeParams
from optv.calibration import Calibration
from optv.tracking_framebuf import read_targets, TargetArray
from optv.correspondences import MatchedCoords, correspondences
from optv.imgcoord import image_coordinates
from optv.transforms import convert_arr_metric_to_pixel

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
        self.assertTrue(np.all(pos[1:,0] > pos[:-1,0]))
        
        # Manually verified order for the loaded data:
        np.testing.assert_array_equal(
            pnr, np.r_[6, 11, 10,  8,  1,  4,  7,  0,  2,  9,  5,  3, 12])
        
class TestCorresp(unittest.TestCase):
    def test_full_corresp(self):
        """Full scene correspondences"""
        cpar = ControlParams(4)
        cpar.read_control_par("testing_fodder/corresp/control.par")
        vpar = VolumeParams()
        vpar.read_volume_par("testing_fodder/corresp/criteria.par")
        
        # Cameras are at so high angles that opposing cameras don't see each 
        # other in the normal air-glass-water setting.
        cpar.get_multimedia_params().set_layers([1.0001], [1.])
        cpar.get_multimedia_params().set_n3(1.0001)
        
        cals = []
        img_pts = []
        corrected = []
        for c in range(4):
            cal = Calibration()
            cal.from_file(
                "testing_fodder/calibration/sym_cam%d.tif.ori" % (c + 1),
                "testing_fodder/calibration/cam1.tif.addpar")
            cals.append(cal)
        
            # Generate test targets.
            targs = TargetArray(16)
            for row, col in np.ndindex(4, 4):
                targ_ix = row*4 + col
                # Avoid symmetric case:
                if (c % 2):
                    targ_ix = 15 - targ_ix
                targ = targs[targ_ix]
                
                pos3d = 10*np.array([[col, row, 0]], dtype=np.float64)
                pos2d = image_coordinates(
                    pos3d, cal, cpar.get_multimedia_params())
                targ.set_pos(convert_arr_metric_to_pixel(pos2d, cpar)[0])
                
                targ.set_pnr(targ_ix)
                targ.set_pixel_counts(25, 5, 5)
                targ.set_sum_grey_value(10)
            
            img_pts.append(targs)
            mc = MatchedCoords(targs, cpar, cal)
            corrected.append(mc)
        
        _, _, num_targs = correspondences(
            img_pts, corrected, cals, vpar, cpar)
        self.assertEqual(num_targs, 16)

    def test_single_cam_corresp(self):
        """Single camera correspondence"""
        cpar = ControlParams(1)
        cpar.read_control_par("testing_fodder/single_cam/parameters/ptv.par")
        vpar = VolumeParams()
        vpar.read_volume_par("testing_fodder/single_cam/parameters/criteria.par")
        
        # Cameras are at so high angles that opposing cameras don't see each 
        # other in the normal air-glass-water setting.
        cpar.get_multimedia_params().set_layers([1.], [1.])
        cpar.get_multimedia_params().set_n3(1.)
        
        cals = []
        img_pts = []
        corrected = []
        cal = Calibration()
        cal.from_file(
            "testing_fodder/single_cam/calibration/cam_1.tif.ori",
            "testing_fodder/single_cam/calibration/cam_1.tif.addpar")
        cals.append(cal)
        
        # Generate test targets.
        targs = TargetArray(9)
        for row, col in np.ndindex(3, 3):
            targ_ix = row*3 + col
            targ = targs[targ_ix]
            
            pos3d = 10*np.array([[col, row, 0]], dtype=np.float64)
            pos2d = image_coordinates(
                pos3d, cal, cpar.get_multimedia_params())
            targ.set_pos(convert_arr_metric_to_pixel(pos2d, cpar)[0])
            
            targ.set_pnr(targ_ix)
            targ.set_pixel_counts(25, 5, 5)
            targ.set_sum_grey_value(10)
            
        img_pts.append(targs)
        mc = MatchedCoords(targs, cpar, cal)
        corrected.append(mc)
        
        _, _, num_targs = correspondences(
            img_pts, corrected, cals, vpar, cpar)

        self.assertEqual(num_targs, 9)



if __name__ == "__main__":
    import sys, os
    print((os.path.abspath(os.curdir)))
    unittest.main()