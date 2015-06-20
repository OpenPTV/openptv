import unittest
from optv.calibration import Calibration
import numpy, os, filecmp, shutil

class Test_Calibration(unittest.TestCase):
    def test_Calibration_instantiation(self):
        
        input_ori_file_name = "testing_fodder/cal/cam1.tif.ori"
        input_add_file_name = "testing_fodder/cal/cam2.tif.addpar"
        output_directory = "testing_fodder/cal/testing_output/"
        
        test_output_ori_file_name = output_directory + "output_ori"
        test_output_add_file_name = output_directory + "output_add"
        
        # create an instance of Calibration wrapper class
        calib_obj = Calibration()
        
        # read calibration parameters from files
        calib_obj.read_calibration(input_ori_file_name, input_add_file_name)
        
        # create an output directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            
        # write calibration parameters to files
        calib_obj.write_calibration(test_output_ori_file_name, test_output_add_file_name)
        
        # open the input and output files an compare them
        output_ori_file = open(test_output_ori_file_name, "r")
        output_add_file = open(test_output_add_file_name, "r")
        
        input_ori_file = open(test_output_ori_file_name, "r")
        input_add_file = open(input_add_file_name, "r")
        
        # compare files and assert they are the same
        self.assertTrue(filecmp.cmp(input_ori_file_name, test_output_ori_file_name, 0))
        self.assertTrue(filecmp.cmp(input_add_file_name, test_output_add_file_name, 0))
        
        # test set_pos() by passing a numpy array of 3 elements
        new_np = numpy.array([111.1111, 222.2222, 333.3333])
        calib_obj.set_pos(new_np)

        # test getting position and assert that position is equal to set position
        numpy.testing.assert_array_equal(new_np, calib_obj.get_pos())
        
        # assert set_pos() raises ValueError exception when given more or less than 3 elements 
        self.assertRaises(ValueError, calib_obj.set_pos, numpy.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, calib_obj.set_pos, numpy.array([1, 2]))
        
        # set angles and assert the angles were set correctly
        dmatrix_before = calib_obj.get_dmatrix()  # dmatrix before setting angles
        angles_np = numpy.array([0.1111, 0.2222, 0.3333])
        calib_obj.set_angles(angles_np)
        dmatrix_after = calib_obj.get_dmatrix()  # dmatrix after setting angles
        # make sure the angles are as were set  
        numpy.testing.assert_array_equal(calib_obj.get_angles(), angles_np)
        
        # assert dmatrix was recalculated (before vs after)
        self.assertFalse(numpy.array_equal(dmatrix_before, dmatrix_after))
        
        # assert set_angles() raises ValueError exception when given more or less than 3 elements 
        self.assertRaises(ValueError, calib_obj.set_angles, numpy.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, calib_obj.set_angles, numpy.array([1, 2]))
        
        # remove the testing output directory and its files
        shutil.rmtree(output_directory)
        
if __name__ == "__main__":
    unittest.main()
