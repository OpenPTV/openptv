import numpy as np
import random
import unittest

from optv.calibration import Calibration
from optv.imgcoord import image_coordinates
from optv.orientation import match_detection_to_ref
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

        with self.assertRaises(TypeError):
            match_detection_to_ref(cal=self.calibration,
                                   ref_pts=xyz_input,
                                   img_pts=TargetArray(coords_count - 1),  # test wrong length
                                   cparam=self.control)


if __name__ == "__main__":
    unittest.main()
