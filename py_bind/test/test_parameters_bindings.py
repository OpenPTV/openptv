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
        
        arr = m.get_n2(copy=False) # don't copy the values: link directly to memory 
        arr[0] = 77.77
        arr[1] = 88.88
        arr[2] = 99.99
        # assert that the arr affected the contents of m object
        numpy.testing.assert_array_equal(m.get_n2(), [77.77, 88.88, 99.99])

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
        self.seq_obj = SequenceParams(4)
        
    def test_read_sequence(self):
        # Fill the SequenceParams object with parameters from test file
        self.seq_obj.read_sequence_par(self.input_sequence_par_file_name, 4)
        
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
            self.seq_obj.set_img_base_name(cam, newStr)
            self.failUnless(self.seq_obj.get_img_base_name(cam) == newStr)
             
        self.seq_obj.set_first(1234)
        self.failUnless(self.seq_obj.get_first() == 1234)
        self.seq_obj.set_last(5678)
        self.failUnless(self.seq_obj.get_last() == 5678)
       
    # testing __richcmp__ comparison method of SequenceParams class
    def test_rich_compare(self):
        self.seq_obj2 = SequenceParams(4)
        self.seq_obj2.read_sequence_par(self.input_sequence_par_file_name, 4)
        
        self.seq_obj3 = SequenceParams(4)
        self.seq_obj3.read_sequence_par(self.input_sequence_par_file_name, 4)
               
        self.failUnless(self.seq_obj2 == self.seq_obj3)
        self.failIf(self.seq_obj2 != self.seq_obj3)
            
        self.seq_obj2.set_first(-999)
        self.failUnless(self.seq_obj2 != self.seq_obj3)
        self.failIf(self.seq_obj2 == self.seq_obj3)
            
        with self.assertRaises(TypeError):
            var = (self.seq_obj2 > self.seq_obj3)
     
    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.temp_output_directory)
        
class Test_VolumeParams(unittest.TestCase):
    def setUp(self):
        self.input_volume_par_file_name = "testing_fodder/volume_parameters/volume.par"
        self.temp_output_directory = "testing_fodder/volume_parameters/testing_output"
        
        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.temp_output_directory):
            os.makedirs(self.temp_output_directory)
            
        # create an instance of VolumeParams class
        self.vol_obj = VolumeParams()
        
    def test_read_volume(self):
        # Fill the VolumeParams object with parameters from test file
        self.vol_obj.read_volume_par(self.input_volume_par_file_name)
        
        # check that all parameters are equal to the contents of test file
        numpy.testing.assert_array_equal(numpy.array([111.111, 222.222]), self.vol_obj.get_X_lay())
        numpy.testing.assert_array_equal(numpy.array([333.333, 444.444]), self.vol_obj.get_Zmin_lay())
        numpy.testing.assert_array_equal(numpy.array([555.555, 666.666]), self.vol_obj.get_Zmax_lay())
         
        self.failUnless(self.vol_obj.get_cnx() == 777.777)
        self.failUnless(self.vol_obj.get_cny() == 888.888)
        self.failUnless(self.vol_obj.get_cn() == 999.999)
        self.failUnless(self.vol_obj.get_csumg() == 1010.1010)
        self.failUnless(self.vol_obj.get_corrmin() == 1111.1111)
        self.failUnless(self.vol_obj.get_eps0() == 1212.1212)
        
    def test_setters(self):
        xlay = numpy.array([111.1, 222.2])
        self.vol_obj.set_X_lay(xlay)
        numpy.testing.assert_array_equal(xlay, self.vol_obj.get_X_lay())
        
        zmin = numpy.array([333.3, 444.4])
        self.vol_obj.set_Zmin_lay(zmin)
        numpy.testing.assert_array_equal(zmin, self.vol_obj.get_Zmin_lay())
        
        zmax = numpy.array([555.5, 666.6])
        self.vol_obj.set_Zmax_lay(zmax)
        numpy.testing.assert_array_equal(zmax, self.vol_obj.get_Zmax_lay())
        
        self.vol_obj.set_cn(1)
        self.failUnless(self.vol_obj.get_cn() == 1)
        
        self.vol_obj.set_cnx(2)
        self.failUnless(self.vol_obj.get_cnx() == 2)
        
        self.vol_obj.set_cny(3)
        self.failUnless(self.vol_obj.get_cny() == 3)
        
        self.vol_obj.set_csumg(4)
        self.failUnless(self.vol_obj.get_csumg() == 4)
        
        self.vol_obj.set_eps0(5)
        self.failUnless(self.vol_obj.get_eps0() == 5)
        
        self.vol_obj.set_corrmin(6)
        self.failUnless(self.vol_obj.get_corrmin() == 6)
     
    # testing __richcmp__ comparison method of VolumeParams class
    def test_rich_compare(self):
        self.vol_obj2 = VolumeParams()
        self.vol_obj2.read_volume_par(self.input_volume_par_file_name)
        self.vol_obj3 = VolumeParams()
        self.vol_obj3.read_volume_par(self.input_volume_par_file_name)
        self.failUnless(self.vol_obj2 == self.vol_obj3)
        self.failIf(self.vol_obj2 != self.vol_obj3)
        
        self.vol_obj2.set_cn(-999)
        self.failUnless(self.vol_obj2 != self.vol_obj3)
        self.failIf(self.vol_obj2 == self.vol_obj3)
        
        with self.assertRaises(TypeError):
            var = (self.vol_obj2 < self.vol_obj3)
    
    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.temp_output_directory)

class Test_ControlParams(unittest.TestCase):
    def setUp(self):
        self.input_control_par_file_name = "testing_fodder/control_parameters/control.par"
        self.temp_output_directory = "testing_fodder/control_parameters/testing_output"
        
        # create a temporary output directory (will be deleted by the end of test)
        if not os.path.exists(self.temp_output_directory):
            os.makedirs(self.temp_output_directory)
        # create an instance of ControlParams class
        self.cp_obj = ControlParams(4)
        
    def test_read_control(self):
        # Fill the ControlParams object with parameters from test file
        self.cp_obj.read_control_par(self.input_control_par_file_name)
        # check if all parameters are equal to the contents of test file
        self.failUnless(self.cp_obj.get_img_base_name(0) == "dumbbell/cam1_Scene77_4085") 
        self.failUnless(self.cp_obj.get_img_base_name(1) == "dumbbell/cam2_Scene77_4085")
        self.failUnless(self.cp_obj.get_img_base_name(2) == "dumbbell/cam3_Scene77_4085")
        self.failUnless(self.cp_obj.get_img_base_name(3) == "dumbbell/cam4_Scene77_4085")
        
        self.failUnless(self.cp_obj.get_cal_img_base_name(0) == "cal/cam1.tif")
        self.failUnless(self.cp_obj.get_cal_img_base_name(1) == "cal/cam2.tif")
        self.failUnless(self.cp_obj.get_cal_img_base_name(2) == "cal/cam3.tif")
        self.failUnless(self.cp_obj.get_cal_img_base_name(3) == "cal/cam4.tif")
        
        self.failUnless(self.cp_obj.get_num_cams() == 4)
        self.failUnless(self.cp_obj.get_hp_flag() == 10)
        self.failUnless(self.cp_obj.get_allCam_flag() == 11)
        self.failUnless(self.cp_obj.get_tiff_flag() == 12)
        self.failUnless(self.cp_obj.get_imx() == 1280)
        self.failUnless(self.cp_obj.get_imy() == 1024)
        self.failUnless(self.cp_obj.get_pix_x() == 15.15)
        self.failUnless(self.cp_obj.get_pix_y() == 16.16)
        self.failUnless(self.cp_obj.get_chfield() == 17)
        
        self.failUnless(self.cp_obj.get_multimedia_params().get_n1() == 18)
        self.failUnless(self.cp_obj.get_multimedia_params().get_n2()[0] == 19.19)
        self.failUnless(self.cp_obj.get_multimedia_params().get_n3() == 20.20)
        self.failUnless(self.cp_obj.get_multimedia_params().get_d()[0] == 21.21)
     
    def test_getters_setters(self):
        cams_num = 4
        for cam in range(cams_num):
            new_str = str(cam) + "some string" + str(cam)
            
            self.cp_obj.set_img_base_name(cam, new_str)
            self.failUnless(self.cp_obj.get_img_base_name(cam) == new_str)
            
            self.cp_obj.set_cal_img_base_name(cam, new_str)
            self.failUnless(self.cp_obj.get_cal_img_base_name(cam) == new_str)
        
        self.cp_obj.set_hp_flag(1)
        self.failUnless(self.cp_obj.get_hp_flag() == 1)
        
        self.cp_obj.set_allCam_flag(2)
        self.failUnless(self.cp_obj.get_allCam_flag() == 2)
        
        self.cp_obj.set_tiff_flag(3)
        self.failUnless(self.cp_obj.get_tiff_flag() == 3)
        
        self.cp_obj.set_imx(4)
        self.failUnless(self.cp_obj.get_imx() == 4)
        
        self.cp_obj.set_imy(5)
        self.failUnless(self.cp_obj.get_imy() == 5)
        
        self.cp_obj.set_pix_x(6)
        self.failUnless(self.cp_obj.get_pix_x() == 6)
        
        self.cp_obj.set_pix_y(7)
        self.failUnless(self.cp_obj.get_pix_y() == 7)
        
        self.cp_obj.set_chfield(8)
        self.failUnless(self.cp_obj.get_chfield() == 8)
         
    # testing __richcmp__ comparison method of ControlParams class
    def test_rich_compare(self):
        self.cp_obj2 = ControlParams(4)
        self.cp_obj2.read_control_par(self.input_control_par_file_name)
        
        self.cp_obj3 = ControlParams(4)
        self.cp_obj3.read_control_par(self.input_control_par_file_name)
           
        self.failUnless(self.cp_obj2 == self.cp_obj3)
        self.failIf(self.cp_obj2 != self.cp_obj3)
           
        self.cp_obj2.set_hp_flag(-999)
        self.failUnless(self.cp_obj2 != self.cp_obj3)
        self.failIf(self.cp_obj2 == self.cp_obj3)
        
        with self.assertRaises(TypeError):
            var = (self.cp_obj2 > self.cp_obj3)  # unhandled operator > 
      
    def tearDown(self):
        # remove the testing output directory and its files
        shutil.rmtree(self.temp_output_directory)        

if __name__ == "__main__":
    unittest.main()
