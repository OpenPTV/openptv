import numpy as np
import random
import unittest
import os

from optv.calibration import Calibration
from optv.imgcoord import image_coordinates
from optv.orientation import match_detection_to_ref, point_positions
from optv.parameters import ControlParams
from optv.tracking_framebuf import TargetArray
from optv.transforms import convert_arr_metric_to_pixel


class Test_Orientation(unittest.TestCase):
    def setUp(self):
        self.input_ori_file_name = r'testing_fodder/calibration/cam1.tif.ori'
        self.input_add_file_name = r'testing_fodder/calibration/cam2.tif.addpar'
        self.control_file_name = r'testing_fodder/control_parameters/control.par'

        self.calibration = Calibration()
        self.calibration.from_file(self.input_ori_file_name, self.input_add_file_name)
        self.control = ControlParams(4)
        self.control.read_control_par(self.control_file_name)

    def test_match_detection_to_ref(self):
        xyz_input = np.array([(10, 10, 10),
                              (200, 200, 200),
                              (600, 800, 100),
                              (20, 10, 2000),
                              (30, 30, 30)], dtype=float)
        coords_count = len(xyz_input)

        xy_img_pts_metric = image_coordinates(xyz_input, self.calibration, self.control.get_multimedia_params())
        xy_img_pts_pixel = convert_arr_metric_to_pixel(xy_img_pts_metric, control=self.control)

        # convert to TargetArray object
        target_array = TargetArray(coords_count)

        for i in range(coords_count):
            target_array[i].set_pnr(i)
            target_array[i].set_pos((xy_img_pts_pixel[i][0], xy_img_pts_pixel[i][1]))

        # create randomized target array
        indices = range(coords_count)
        shuffled_indices = range(coords_count)

        while indices == shuffled_indices:
            random.shuffle(shuffled_indices)

        randomized_target_array = TargetArray(coords_count)
        for i in range(coords_count):
            randomized_target_array[shuffled_indices[i]].set_pos(target_array[i].pos())
            randomized_target_array[shuffled_indices[i]].set_pnr(target_array[i].pnr())

        # match detection to reference
        matched_target_array = match_detection_to_ref(cal=self.calibration,
                                                      ref_pts=xyz_input,
                                                      img_pts=randomized_target_array,
                                                      cparam=self.control)

        # assert target array is as before
        for i in range(coords_count):
            if matched_target_array[i].pos() != target_array[i].pos() \
                    or matched_target_array[i].pnr() != target_array[i].pnr():
                self.fail('match_detection_to_ref failed to match detection to reference.')

        # pass ref_pts and img_pts with non-equal lengths
        with self.assertRaises(TypeError):
            match_detection_to_ref(cal=self.calibration,
                                   ref_pts=xyz_input,
                                   img_pts=TargetArray(coords_count - 1),
                                   cparam=self.control)

    def test_point_positions(self):
        # prepare MultimediaParams
        mult_params = self.control.get_multimedia_params()

        mult_params.set_n1(1.)
        mult_params.set_layers(np.array([1.]), np.array([1.]))
        mult_params.set_n3(1.)

        # 3d point
        points = np.array([[(17, 42, 0)],
                           [(17, 42, 0)]], dtype=float)

        num_cams = 4
        ori_tmpl = r'testing_fodder/calibration/sym_cam{cam_num}.tif.ori'
        add_file = r'testing_fodder/calibration/cam1.tif.addpar'
        calibs = []
        targs_plain = np.empty([0, 2])
        targs_jigged = np.empty([0, 2])

        jigg_amp = 0.5

        # read calibration for each camera from files
        for cam in range(num_cams):
            ori_name = ori_tmpl.format(cam_num=cam + 1)
            new_cal = Calibration()
            new_cal.from_file(ori_file=ori_name, add_file=add_file)
            calibs.append(new_cal)

        for point in points:
            for cam_cal in calibs:
                new_plain_targ = image_coordinates(point, cam_cal, self.control.get_multimedia_params())
                targs_plain = np.append(targs_plain, new_plain_targ, 0)

                jigged_point = np.copy(point)
                for j in jigged_point:
                    j[1] += (jigg_amp if (calibs.index(cam_cal) % 2) != 0 else -jigg_amp)

                new_jigged_targ = image_coordinates(jigged_point, cam_cal, self.control.get_multimedia_params())
                targs_jigged = np.append(targs_jigged, new_jigged_targ, 0)

            targets_plain = np.empty(shape=(len(point), num_cams, 2))
            targets_jigged = np.empty(shape=(len(point), num_cams, 2))
            for targ_ix in range(len(point)):
                for cam_ix in range(num_cams):
                    for coord_ix in range(2):
                        targets_plain[targ_ix][cam_ix][coord_ix] = targs_plain[cam_ix][coord_ix]
                        targets_jigged[targ_ix][cam_ix][coord_ix] = targs_jigged[cam_ix][coord_ix]

            skew_dist_plain = point_positions(targets=targets_plain, cparam=self.control, cals=calibs)
            skew_dist_jigged = point_positions(targets=targets_jigged, cparam=self.control, cals=calibs)

            for targ_ix in range(len(targets_plain)):
                if skew_dist_plain[1][targ_ix] > 1e-10:
                    self.fail('skew distance of target#{targ_num} is more than allowed'.format(targ_num=targ_ix + 1))

            if np.linalg.norm(np.subtract(point, skew_dist_plain[0])) > 1e-10:
                self.fail('matrix norm for skew distance is bigger than allowed')

            jigged_correct = 4 * (2 * jigg_amp) / 6
            if abs(skew_dist_jigged[1] - jigged_correct) > 0.05:
                self.fail('jigged skew distance is bigger than allowed')

            if np.linalg.norm(np.subtract(point, skew_dist_jigged[0])) > 0.01:
                self.fail('matrix norm for skew distance is bigger than allowed')


if __name__ == "__main__":
    unittest.main()
