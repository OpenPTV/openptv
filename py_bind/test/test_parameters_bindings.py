import unittest
from optv.parameters import *
import numpy, os, filecmp, shutil

class Test_MultimediaParams(unittest.TestCase):
    def test_mm_np_instantiation(self):
        
        n2_np = numpy.array([11, 22, 33])
        d_np = numpy.array([55, 66, 77])
        
        m = MultimediaParams(nlay=3, n1=2, n2=n2_np, d=d_np, n3=4, lut=1)
        
        self.failUnlessEqual(m.get_nlay(), 3)
        self.failUnlessEqual(m.get_n1(), 2)
        self.failUnlessEqual(m.get_n3(), 4)
        self.failUnlessEqual(m.get_lut(), 1)
        
        numpy.testing.assert_array_equal(m.get_d(), d_np)
        numpy.testing.assert_array_equal(m.get_n2(), n2_np)
        
        self.failUnlessEqual(m.__str__(), "nlay=\t3 \nn1=\t2.0 \nn2=\t{11.0, 22.0, 33.0} \nd=\t{55.0, 66.0, 77.0} \nn3=\t4.0 \nlut=\t1 ")

class Test_TrackingParams(unittest.TestCase):
    
    def setUp(self):
        self.input_tracking_par_file_name = "testing_fodder/tracking_parameters/track.par"
        self.output_directory = "testing_fodder/tracking_parameters/testing_output"
        
        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        # create an instance of TrackingParams class
        # testing setters that are used in constructor
        self.track_obj1 = TrackingParams(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9, 10, 11, 12, 13)
        
    # Testing getters according to the values passed in setUp
    
    def test_TrackingParams_getters(self):
        self.failUnless(self.track_obj1.get_dacc() == 1.1)
        self.failUnless(self.track_obj1.get_dangle() == 2.2)
        self.failUnless(self.track_obj1.get_dvxmax() == 3.3)
        self.failUnless(self.track_obj1.get_dvxmin() == 4.4)
        self.failUnless(self.track_obj1.get_dvymax() == 5.5)
        self.failUnless(self.track_obj1.get_dvymin() == 6.6)
        self.failUnless(self.track_obj1.get_dvzmax() == 7.7)
        self.failUnless(self.track_obj1.get_dvzmin() == 8.8)
        self.failUnless(self.track_obj1.get_dsumg() == 9)
        self.failUnless(self.track_obj1.get_dn() == 10)
        self.failUnless(self.track_obj1.get_dnx() == 11)
        self.failUnless(self.track_obj1.get_dny() == 12)
        self.failUnless(self.track_obj1.get_add() == 13)

    def test_TrackingParams_read_from_file(self):
        """Filling a TrackingParams object by reading file"""
        
        # read tracking parameters from file
        self.track_obj1.read_track_par(self.input_tracking_par_file_name)
        
        # check that the values of track_obj1 are equal to values in tracking parameters file
        # the check is performed according to the order the parameters were read from same file
        track_file = open(self.input_tracking_par_file_name, 'r')
        self.failUnless(self.track_obj1.get_dvxmin() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dvxmax() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dvymin() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dvymax() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dvzmin() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dvzmax() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dangle() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_dacc() == float(track_file.readline()))
        self.failUnless(self.track_obj1.get_add() == int(track_file.readline()))
    
        self.failUnless(self.track_obj1.get_dsumg() == 0)
        self.failUnless(self.track_obj1.get_dn() == 0)
        self.failUnless(self.track_obj1.get_dnx() == 0)
        self.failUnless(self.track_obj1.get_dny() == 0)
    def test_comparison(self):
        # create two identical objects
        self.track_obj2 = TrackingParams(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9, 10, 11, 12, 13)
        self.track_obj3 = TrackingParams(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9, 10, 11, 12, 13)
        
        self.failUnless(self.track_obj2 == self.track_obj3)
        self.failIf(self.track_obj2 != self.track_obj3)
        
        # change one instance variable of track_obj2 and check that the comparisons results are inverted
        # please note that the operands '==' and '!=' must be checked separately
        self.track_obj2.set_dvxmin(999.999)
        self.failUnless(self.track_obj2 != self.track_obj3)
        self.failIf(self.track_obj2 == self.track_obj3)
        
#         get_dacc()
#         get_dangle()
#         get_dvxmax():
#         get_dvxmin():
#         get_dvymax():
#         get_dvymin():
#         get_dvzmax():
#         get_dvzmin():
#         get_dsumg():
#         get_dn():
#         get_dnx():
#         get_dny():
#         get_add():
#         
#     def test_set_pos(self):
#         """Set exterior position, only for admissible values"""
#         # test set_pos() by passing a numpy array of 3 elements
#         new_np = numpy.array([111.1111, 222.2222, 333.3333])
#         self.cal.set_pos(new_np)
# 
#         # test getting position and assert that position is equal to set position
#         numpy.testing.assert_array_equal(new_np, self.cal.get_pos())
#         
#         # assert set_pos() raises ValueError exception when given more or less than 3 elements 
#         self.assertRaises(ValueError, self.cal.set_pos, numpy.array([1, 2, 3, 4]))
#         self.assertRaises(ValueError, self.cal.set_pos, numpy.array([1, 2]))
#     
#     def test_set_angles(self):
#         # set angles and assert the angles were set correctly
#         dmatrix_before = self.cal.get_rotation_matrix()  # dmatrix before setting angles
#         angles_np = numpy.array([0.1111, 0.2222, 0.3333])
#         self.cal.set_angles(angles_np)
#         dmatrix_after = self.cal.get_rotation_matrix()  # dmatrix after setting angles
#         # make sure the angles are as were set  
#         numpy.testing.assert_array_equal(self.cal.get_angles(), angles_np)
#         
#         # assert dmatrix was recalculated (before vs after)
#         self.assertFalse(numpy.array_equal(dmatrix_before, dmatrix_after))
#         
#         # assert set_angles() raises ValueError exception when given more or less than 3 elements 
#         self.assertRaises(ValueError, self.cal.set_angles, numpy.array([1, 2, 3, 4]))
#         self.assertRaises(ValueError, self.cal.set_angles, numpy.array([1, 2]))
#     
#     def tearDown(self):
#         # remove the testing output directory and its files
#         shutil.rmtree(self.output_directory)
        
if __name__ == "__main__":
    unittest.main()
