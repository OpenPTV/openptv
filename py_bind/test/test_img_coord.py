import unittest
from optv.imgcoord import flat_image_coordinates, image_coordinates
from optv.parameters import ControlParams, MultimediaParams
from optv.calibration import Calibration

import numpy as np

class Test_image_coordinates(unittest.TestCase):
    def setUp(self):
        self.control = ControlParams(4) 
    
        self.calibration = Calibration()
        
    def test_img_coord_typecheck(self):
        
        with self.assertRaises(TypeError):
            list = [[0 for x in range(3)] for x in range(10)]  # initialize a 10x3 list (but not numpy matrix)
            flat_image_coordinates(list, self.control, out=None)
        with self.assertRaises(TypeError):
            flat_image_coordinates(np.empty((10, 2)), self.calibration, self.control.get_multimedia_params(), output=None)
        with self.assertRaises(TypeError):
            image_coordinates(np.empty((10, 3)), self.calibration, self.control.get_multimedia_params(), output=np.zeros((10, 3)))
        with self.assertRaises(TypeError):
            image_coordinates(np.zeros((10, 2)), self.calibration, self.control.get_multimedia_params(), output=np.zeros((10, 2)))
   
    def test_image_coord_regress(self):
        
        self.calibration.set_pos(np.array([0, 0, 40]))
        self.calibration.set_angles(np.array([0, 0, 0]))
        self.calibration.set_primary_point(np.array([0, 0, 10]))
        self.calibration.set_glass_vec(np.array([0, 0, 20]))
        self.calibration.set_radial_distortion(np.array([0, 0, 0]))
        self.calibration.set_decentering(np.array([0, 0]))
        self.calibration.set_affine_trans(np.array([1, 0]))

        self.mult = MultimediaParams(n1=1,
                                     n2=np.array([1]),
                                     n3=1,
                                     d=np.array([1]))
        
        input = np.array([[10., 5., -20.],
                          [10., 5., -20.]])  # vec3d
        output = np.zeros((2, 2))
        
        x = 10. / 6.
        y = x / 2.
        correct_output = np.array([[x, y],
                                   [x, y]])

        flat_image_coordinates(input=input, cal=self.calibration, mult_params=self.mult, output=output)
        np.testing.assert_array_equal(output, correct_output)
        
        output=np.full((2,2), 999.)
        image_coordinates(input=input, cal=self.calibration, mult_params=self.mult, output=output)

        np.testing.assert_array_equal(output, correct_output)
        
if __name__ == '__main__':
  unittest.main()
 
