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
        

    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.output_directory)
        
class Test_SequenceParams(unittest.TestCase):
    def setUp(self):
        
        self.input_sequence_par_file_name = "testing_fodder/sequence_parameters/sequence.par"
        self.temp_output_directory = "testing_fodder/sequence_parameters/testing_output"
        
        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.temp_output_directory):
            os.makedirs(self.temp_output_directory)
            
        # create an instance of SequencParams class
        self.seq_obj = SequenceParams()
        
    def test_read_sequence(self):
        # Fill the SequenceParams object with parameters from test file
        self.seq_obj.read_sequence_par(self.input_sequence_par_file_name)
        
        # check that all parameters are equal to the contents of test file
        self.failUnless(self.seq_obj.get_img_base_name(0) == "dumbbell/cam1_Scene77_") 
        self.failUnless(self.seq_obj.get_img_base_name(1) == "dumbbell/cam2_Scene77_")
        self.failUnless(self.seq_obj.get_img_base_name(2) == "dumbbell/cam3_Scene77_")
        self.failUnless(self.seq_obj.get_img_base_name(3) == "dumbbell/cam4_Scene77_")
        self.failUnless(self.seq_obj.get_first() == 497)
        self.failUnless(self.seq_obj.get_last() == 597)
    
    def test_getters_setters(self):
        cams_num = 4
        for cam in range(cams_num):
            newStr = str(cam) + "some string" + str(cam)
            # print "going to set in imgname for cam#" +str(cam)+":\t\t"+newStr
            self.seq_obj.set_img_base_name(cam, newStr)
            # print "got from get_img_name for cam#"+str(cam) +":\t\t"+self.seq_obj.get_img_base_name(cam)
            self.failUnless(self.seq_obj.get_img_base_name(cam) == newStr)
        
        self.seq_obj.set_first(1234)
        self.failUnless(self.seq_obj.get_first() == 1234)
        self.seq_obj.set_last(5678)
        self.failUnless(self.seq_obj.get_last() == 5678)
        
    
    def tearDown(self):
       
        # remove the testing output directory and its files
        shutil.rmtree(self.temp_output_directory)
        
if __name__ == "__main__":
    unittest.main()
