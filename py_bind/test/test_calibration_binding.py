import unittest
from optv.calibration import Calibration
import numpy, os, filecmp, shutil

class Test_Calibration(unittest.TestCase):
    def setUp(self):        
        self.input_ori_file_name = "testing_fodder/calibration/cam1.tif.ori"
        self.input_add_file_name = "testing_fodder/calibration/cam2.tif.addpar"
        self.output_directory = "testing_fodder/calibration/testing_output/"
        
        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        # create an instance of Calibration wrapper class
        self.cal = Calibration()
            
    def test_Calibration_instantiation(self):
        """Filling a calibration object by reading ori files"""
        self.output_ori_file_name = self.output_directory + "output_ori"
        self.output_add_file_name = self.output_directory + "output_add"
                
        # read calibration parameters from files
        self.cal.read_calibration(self.input_ori_file_name, self.input_add_file_name)
            
        # write calibration parameters to files
        self.cal.write_calibration(self.output_ori_file_name, self.output_add_file_name)
        
        # Compare input and output files and assert they are the same
        self.assertTrue(filecmp.cmp(self.input_ori_file_name, self.output_ori_file_name, 0))
        self.assertTrue(filecmp.cmp(self.input_add_file_name, self.output_add_file_name, 0))
        
    def test_set_pos(self):
        """Set exterior position, only for admissible values"""
        # test set_pos() by passing a numpy array of 3 elements
        new_np = numpy.array([111.1111, 222.2222, 333.3333])
        self.cal.set_pos(new_np)

        # test getting position and assert that position is equal to set position
        numpy.testing.assert_array_equal(new_np, self.cal.get_pos())
        
        # assert set_pos() raises ValueError exception when given more or less than 3 elements 
        self.assertRaises(ValueError, self.cal.set_pos, numpy.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, self.cal.set_pos, numpy.array([1, 2]))
    
    def test_set_angles(self):
        # set angles and assert the angles were set correctly
        dmatrix_before = self.cal.get_rotation_matrix()  # dmatrix before setting angles
        angles_np = numpy.array([0.1111, 0.2222, 0.3333])
        self.cal.set_angles(angles_np)
        dmatrix_after = self.cal.get_rotation_matrix()  # dmatrix after setting angles
        # make sure the angles are as were set  
        numpy.testing.assert_array_equal(self.cal.get_angles(), angles_np)
        
        # assert dmatrix was recalculated (before vs after)
        self.assertFalse(numpy.array_equal(dmatrix_before, dmatrix_after))
        
        # assert set_angles() raises ValueError exception when given more or less than 3 elements 
        self.assertRaises(ValueError, self.cal.set_angles, numpy.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, self.cal.set_angles, numpy.array([1, 2]))
    
    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.output_directory)
        
if __name__ == "__main__":
    unittest.main()
