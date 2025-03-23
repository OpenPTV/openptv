import numpy as np
import random
import unittest

from optv.calibration import Calibration
from optv.imgcoord import image_coordinates, flat_image_coordinates
from optv.orientation import match_detection_to_ref, point_positions, \
    external_calibration, full_calibration, dumbbell_target_func
from optv.parameters import ControlParams, VolumeParams
from optv.tracking_framebuf import TargetArray
from optv.transforms import convert_arr_metric_to_pixel


class Test_Orientation(unittest.TestCase):
    def setUp(self):
        self.input_ori_file_name = 'testing_fodder/calibration/cam1.tif.ori'
        self.input_add_file_name = 'testing_fodder/calibration/cam2.tif.addpar'
        self.control_file_name = 'testing_fodder/control_parameters/control.par'
        self.volume_file_name = 'testing_fodder/corresp/criteria.par'

        self.calibration = Calibration()
        self.calibration.from_file(
            self.input_ori_file_name, self.input_add_file_name)
        self.control = ControlParams(4)
        self.control.read_control_par(self.control_file_name)
        self.vpar = VolumeParams()
        self.vpar.read_volume_par(self.volume_file_name)

    def test_match_detection_to_ref(self):
        """Match detection to reference (sortgrid)"""
        xyz_input = np.array([(10, 10, 10),
                              (200, 200, 200),
                              (600, 800, 100),
                              (20, 10, 2000),
                              (30, 30, 30)], dtype=float)
        coords_count = len(xyz_input)

        xy_img_pts_metric = image_coordinates(
            xyz_input, self.calibration, self.control.get_multimedia_params())
        xy_img_pts_pixel = convert_arr_metric_to_pixel(
            xy_img_pts_metric, control=self.control)

        # convert to TargetArray object
        target_array = TargetArray(coords_count)

        for i in range(coords_count):
            target_array[i].set_pnr(i)
            target_array[i].set_pos(
                (xy_img_pts_pixel[i][0], xy_img_pts_pixel[i][1]))

        # create randomized target array
        indices = list(range(coords_count))
        shuffled_indices = list(range(coords_count))

        while indices == shuffled_indices:
            random.shuffle(shuffled_indices)

        rand_targ_array = TargetArray(coords_count)
        for i in range(coords_count):
            rand_targ_array[shuffled_indices[i]].set_pos(target_array[i].pos())
            rand_targ_array[shuffled_indices[i]].set_pnr(target_array[i].pnr())

        # match detection to reference
        matched_target_array = match_detection_to_ref(cal=self.calibration,
                                                      ref_pts=xyz_input,
                                                      img_pts=rand_targ_array,
                                                      cparam=self.control)

        # assert target array is as before
        for i in range(coords_count):
            if matched_target_array[i].pos() != target_array[i].pos() \
                    or matched_target_array[i].pnr() != target_array[i].pnr():
                self.fail()

        # # pass ref_pts and img_pts with non-equal lengths
        # with self.assertRaises(ValueError):
        #     match_detection_to_ref(cal=self.calibration,
        #                            ref_pts=xyz_input,
        #                            img_pts=TargetArray(coords_count - 1),
        #                            cparam=self.control)

    def test_point_positions(self):
        """Point positions"""
        # prepare MultimediaParams
        mult_params = self.control.get_multimedia_params()

        mult_params.set_n1(1.)
        mult_params.set_layers(np.array([1.]), np.array([1.]))
        mult_params.set_n3(1.)

        # 3d point
        points = np.array([[17, 42, 0],
                           [17, 42, 0]], dtype=float)
        
        num_cams = 4
        ori_tmpl = 'testing_fodder/calibration/sym_cam{cam_num}.tif.ori'
        add_file = 'testing_fodder/calibration/cam1.tif.addpar'
        calibs = []
        targs_plain = []
        targs_jigged = []

        jigg_amp = 0.5

        # read calibration for each camera from files
        for cam in range(num_cams):
            ori_name = ori_tmpl.format(cam_num=cam + 1).encode()
            new_cal = Calibration()
            new_cal.from_file(ori_file=ori_name, add_file=add_file)
            calibs.append(new_cal)

        for cam_num, cam_cal in enumerate(calibs):
            new_plain_targ = image_coordinates(
                points, cam_cal, self.control.get_multimedia_params())
            targs_plain.append(new_plain_targ)
            
            if (cam_num % 2) == 0:
                jigged_points = points - np.r_[0, jigg_amp, 0]
            else:
                jigged_points = points + np.r_[0, jigg_amp, 0]

            new_jigged_targs = image_coordinates(
                jigged_points, cam_cal, self.control.get_multimedia_params())
            targs_jigged.append(new_jigged_targs)

        targs_plain = np.array(targs_plain).transpose(1,0,2)
        targs_jigged = np.array(targs_jigged).transpose(1,0,2)
        skew_dist_plain = point_positions(targs_plain, self.control, calibs, self.vpar)
        skew_dist_jigged = point_positions(targs_jigged, self.control, calibs, self.vpar)

        if np.any(skew_dist_plain[1] > 1e-10):
            self.fail(('skew distance of target#{targ_num} ' \
                + 'is more than allowed').format(
                    targ_num=np.nonzero(skew_dist_plain[1] > 1e-10)[0][0]))

        if np.any(np.linalg.norm(points - skew_dist_plain[0], axis=1) > 1e-6):
            self.fail('Rays converge on wrong position.')

        if np.any(skew_dist_jigged[1] > 0.7):
            self.fail(('skew distance of target#{targ_num} ' \
                + 'is more than allowed').format(
                    targ_num=np.nonzero(skew_dist_jigged[1] > 1e-10)[0][0]))
        if np.any(np.linalg.norm(points - skew_dist_jigged[0], axis=1) > 0.1):
            self.fail('Rays converge on wrong position after jigging.')

    def test_single_camera_point_positions(self):
        """Point positions for a single camera case"""

        num_cams = 1
        # prepare MultimediaParams
        cpar_file = 'testing_fodder/single_cam/parameters/ptv.par'
        vpar_file = 'testing_fodder/single_cam/parameters/criteria.par'
        cpar = ControlParams(num_cams)
        cpar.read_control_par(cpar_file)
        mult_params = cpar.get_multimedia_params()

        vpar = VolumeParams()
        vpar.read_volume_par(vpar_file)

        ori_name = 'testing_fodder/single_cam/calibration/cam_1.tif.ori'
        add_name = 'testing_fodder/single_cam/calibration/cam_1.tif.addpar'
        calibs = []


        # read calibration for each camera from files
        new_cal = Calibration()
        new_cal.from_file(ori_file=ori_name, add_file=add_name)
        calibs.append(new_cal)


        # 3d point
        points = np.array([[1, 1, 0],
                           [-1, -1, 0]], dtype=float)

        targs_plain = []
        targs_jigged = []


        jigg_amp = 0.4


        new_plain_targ = image_coordinates(
            points, calibs[0], mult_params)
        targs_plain.append(new_plain_targ)

        jigged_points = points - np.r_[0, jigg_amp, 0]

        new_jigged_targs = image_coordinates(
            jigged_points, calibs[0], mult_params)
        targs_jigged.append(new_jigged_targs)

        targs_plain = np.array(targs_plain).transpose(1,0,2)
        targs_jigged = np.array(targs_jigged).transpose(1,0,2)
        skew_dist_plain = point_positions(targs_plain, cpar, calibs, vpar)
        skew_dist_jigged = point_positions(targs_jigged, cpar, calibs, vpar)

        if np.any(np.linalg.norm(points - skew_dist_plain[0], axis=1) > 1e-6):
            self.fail('Rays converge on wrong position.')

        if np.any(np.linalg.norm(jigged_points - skew_dist_jigged[0], axis=1) > 1e-6):
            self.fail('Rays converge on wrong position after jigging.')
    
    def test_dumbbell(self):
        # prepare MultimediaParams
        mult_params = self.control.get_multimedia_params()
        mult_params.set_n1(1.)
        mult_params.set_layers(np.array([1.]), np.array([1.]))
        mult_params.set_n3(1.)

        # 3d point
        points = np.array([[17.5, 42, 0],
                           [-17.5, 42, 0]], dtype=float)
        
        num_cams = 4
        ori_tmpl = 'testing_fodder/dumbbell/cam{cam_num}.tif.ori'
        add_file = 'testing_fodder/calibration/cam1.tif.addpar'
        calibs = []
        targs_plain = []

        # read calibration for each camera from files
        for cam in range(num_cams):
            ori_name = ori_tmpl.format(cam_num=cam + 1)
            new_cal = Calibration()
            new_cal.from_file(ori_file=ori_name.encode(), add_file=add_file.encode())
            calibs.append(new_cal)

        for cam_cal in calibs:
            new_plain_targ = flat_image_coordinates(
                points, cam_cal, self.control.get_multimedia_params())
            targs_plain.append(new_plain_targ)

        targs_plain = np.array(targs_plain).transpose(1,0,2)
        
        # The cameras are not actually fully calibrated, so the result is not 
        # an exact 0. The test is that changing the expected distance changes 
        # the measure.
        tf = dumbbell_target_func(targs_plain, self.control, calibs, 35., 0.)
        self.assertAlmostEqual(tf, 7.14860, 5) # just a regression test
        
        # As we check the db length, the measure increases...
        tf_len = dumbbell_target_func(
            targs_plain, self.control, calibs, 35., 1.)
        self.assertTrue(tf_len > tf)
        
        # ...but not as much as when giving the wrong length.
        tf_too_long = dumbbell_target_func(
            targs_plain, self.control, calibs, 25., 1.)
        self.assertTrue(tf_too_long > tf_len > tf)
        

class TestGradientDescent(unittest.TestCase):
    # Based on the C tests in liboptv/tests/check_orientation.c
    
    def setUp(self):
        control_file_name = 'testing_fodder/corresp/control.par'
        self.control = ControlParams(4)
        self.control.read_control_par(control_file_name)
        
        self.cal = Calibration()
        self.cal.from_file(
            "testing_fodder/calibration/cam1.tif.ori", 
            "testing_fodder/calibration/cam1.tif.addpar")
        self.orig_cal = Calibration()
        self.orig_cal.from_file(
            "testing_fodder/calibration/cam1.tif.ori", 
            "testing_fodder/calibration/cam1.tif.addpar")
    
    def test_external_calibration(self):
        """External calibration using clicked points."""
        ref_pts = np.array([
            [-40., -25., 8.],
            [ 40., -15., 0.],
            [ 40.,  15., 0.],
            [ 40.,   0., 8.]])
    
        # Fake the image points by back-projection
        targets = convert_arr_metric_to_pixel(image_coordinates(
            ref_pts, self.cal, self.control.get_multimedia_params()),
            self.control)
        
        # Jigg the fake detections to give raw_orient some challenge.
        targets[:,1] -= 0.1
        
        self.assertTrue(external_calibration(
            self.cal, ref_pts, targets, self.control))
        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(),
            decimal=4)
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(),
            decimal=3)
    
    def test_full_calibration(self):
        ref_pts = np.array([a.flatten() for a in np.meshgrid(
            np.r_[-60:-30:4j], np.r_[0:15:4j], np.r_[0:15:4j])]).T
        
        # Fake the image points by back-projection
        targets = convert_arr_metric_to_pixel(image_coordinates(
            ref_pts, self.cal, self.control.get_multimedia_params()),
            self.control)
        
        # Full calibration works with TargetArray objects, not NumPy.
        target_array = TargetArray(len(targets))
        for i in range(len(targets)):
            target_array[i].set_pnr(i)
            target_array[i].set_pos(targets[i])
        
        # Perturb the calibration object, then compore result to original.
        self.cal.set_pos(self.cal.get_pos() + np.r_[15., -15., 15.])
        self.cal.set_angles(self.cal.get_angles() + np.r_[-.5, .5, -.5])
        
        _, _, _ = full_calibration(
            self.cal, ref_pts, target_array, self.control)
        
        np.testing.assert_array_almost_equal(
            self.cal.get_angles(), self.orig_cal.get_angles(),
            decimal=4)
        np.testing.assert_array_almost_equal(
            self.cal.get_pos(), self.orig_cal.get_pos(),
            decimal=3)
        
if __name__ == "__main__":
    unittest.main()
