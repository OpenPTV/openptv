import unittest
from optv.parameters import *
from optv.trafo import *
import numpy as np, os, filecmp, shutil

class Test_trafo(unittest.TestCase):
    
    def setUp(self):
        self.input_control_par_file_name = "testing_fodder/control_parameters/control.par"
#         self.temp_output_directory = "testing_fodder/control_parameters/testing_output"
#         
#         # create a temporary output directory (will be deleted by the end of test)
#         if not os.path.exists(self.temp_output_directory):
#             os.makedirs(self.temp_output_directory)
#         # create an instance of ControlParams class
        self.cp_obj = ControlParams(4)
        
#     def test_read_control(self):
#         # Fill the ControlParams object with parameters from test file
        self.cp_obj.read_control_par(self.input_control_par_file_name)
#         print self.cp_obj.get_imx()
        
        # Assert TypeError is raised when passing a non (n,2) shaped numpy ndarray
        with self.assertRaises(TypeError):
            list=[[0 for x in range(2)] for x in range(10)] # initialize a 10x2 list (but not numpy matrix)
            convert_pixel_to_metric(list, self.cp_obj, out=None)
        with self.assertRaises(TypeError):
            convert_pixel_to_metric(np.empty((1,1)), self.cp_obj, out=None)
        with self.assertRaises(TypeError):
            convert_pixel_to_metric(np.empty((1,3)), self.cp_obj, out=None)
       
        shape=(10,2)
        arr=np.full(shape, 1000)
        arr2=np.zeros(shape)
        print arr
        newarr=convert_pixel_to_metric(arr, self.cp_obj, out=arr2)

        print arr
        print arr2
        print newarr
        
    def test_go(self):
        pass
    
if __name__ == '__main__':
  unittest.main()
 