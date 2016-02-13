import unittest
from optv.parameters import *
from optv.transforms import *
import numpy as np, os, filecmp, shutil

class Test_transforms(unittest.TestCase):
    
    def setUp(self):
        self.input_control_par_file_name = "testing_fodder/control_parameters/control.par"
        self.control = ControlParams(4)      
        self.control.read_control_par(self.input_control_par_file_name)
        
    def test_go(self):
        # Assert TypeError is raised when passing a non (n,2) shaped numpy ndarray
        with self.assertRaises(TypeError):
            list = [[0 for x in range(2)] for x in range(10)]  # initialize a 10x2 list (but not numpy matrix)
            convert_arr_pixel_to_metric(list, self.control, out=None)
        with self.assertRaises(TypeError):
            convert_arr_pixel_to_metric(np.empty((10, 3)), self.control, out=None)
        with self.assertRaises(TypeError):
            convert_arr_pixel_to_metric(np.empty((2, 1)), self.control, out=None)
        with self.assertRaises(TypeError):
            convert_arr_pixel_to_metric(np.zeros((11, 2)), self.control, out=np.zeros((12, 2)))
        
        input = np.full((3, 2), 100)
        output = np.zeros((3, 2))
        correct_output_pixel_to_metric = [[-8181.  ,  6657.92],
                                          [-8181.  ,  6657.92],
                                          [-8181.  ,  6657.92]]
        correct_output_metric_to_pixel= [[ 646.60066007,  505.81188119],
                                         [ 646.60066007,  505.81188119],
                                         [ 646.60066007,  505.81188119]]
        
        # Test when passing an array for output
        convert_arr_pixel_to_metric(input, self.control, out=output)
        np.testing.assert_array_almost_equal(output, correct_output_pixel_to_metric,decimal=7)
        output = np.zeros((3, 2))
        convert_arr_metric_to_pixel(input, self.control, out=output)
        np.testing.assert_array_almost_equal(output, correct_output_metric_to_pixel, decimal=7)
        
         # Test when NOT passing an array for output
        output=convert_arr_pixel_to_metric(input, self.control, out=None)
        np.testing.assert_array_almost_equal(output, correct_output_pixel_to_metric,decimal=7)
        output = np.zeros((3, 2))
        output=convert_arr_metric_to_pixel(input, self.control, out=None)
        np.testing.assert_array_almost_equal(output, correct_output_metric_to_pixel, decimal=7)

            
if __name__ == '__main__':
  unittest.main()
 
