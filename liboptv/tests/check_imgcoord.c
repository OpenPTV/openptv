/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"
#include "ray_tracing.h"
#include "multimed.h"
#include "imgcoord.h"
#include <unistd.h>

int file_exists(char *filename);

#define EPS 1E-6


START_TEST(test_img_xy_mm_geo)
{

    double X = 50.0, Y = 100.0, Z = -10.0, x,y ;
    int i_cam = 0;
            
    Calibration *cal;

    
    char ori_file[] = "testing_fodder/cal/cam2.tif.ori";
    char add_file[] = "testing_fodder/cal/cam2.tif.addpar";
    
    ck_assert_msg (file_exists(ori_file) == 1, "\n File %s does not exist\n", ori_file);
    ck_assert_msg (file_exists(add_file) == 1, "\n File %s does not exist\n", add_file);
    cal = read_calibration(ori_file, add_file, NULL);    
    fail_if (cal == NULL, "\n ORI or ADDPAR file reading failed \n");
    

         
    volume_par *vpar;
    char vol_file[] = "testing_fodder/parameters/criteria.par";
    ck_assert_msg (file_exists(vol_file) == 1, "\n File %s does not exist\n", vol_file);    
    vpar = read_volume_par(vol_file);
    fail_if (vpar == NULL, "\n volume parameter file reading failed \n");
    
    
    
    control_par *cpar;
    char filename[] = "testing_fodder/parameters/ptv.par";
    ck_assert_msg (file_exists(filename) == 1, "\n File %s does not exist\n", filename);
    cpar = read_control_par(filename);
    fail_if (cpar == NULL, "\n control parameter file reading failed\n ");
    
    cpar->mm->lut = 0; // to start LUT initialization 
    cpar->num_cams = 1; // only one camera test

    mmlut test_mmlut[4];
    
    init_mmLUT (vpar, cpar, cal, test_mmlut);
    
    img_xy_mm_geo (X,Y,Z, cal, cpar->mm, i_cam, test_mmlut, &x, &y);
    
    ck_assert_msg(  fabs(x + 1.11895292) < EPS && 
                    fabs(y - 33.91358203)  < EPS,
     "Expected -1.11895292 33.91358203  but found %10.8f %10.8f\n", 
     x,y);      
    
}
END_TEST



START_TEST(test_imgcoord)
{
    double X = 100.0, Y = 100.0, Z = 0.0, x,y ;
    int i_cam = 0;
            
    Calibration *cal;

    
    char ori_file[] = "testing_fodder/cal/cam1.tif.ori";
    char add_file[] = "testing_fodder/cal/cam1.tif.addpar";
    
    ck_assert_msg (file_exists(ori_file) == 1, "\n File %s does not exist\n", ori_file);
    ck_assert_msg (file_exists(add_file) == 1, "\n File %s does not exist\n", add_file);
    cal = read_calibration(ori_file, add_file, NULL);    
    fail_if (cal == NULL, "\n ORI or ADDPAR file reading failed \n");
    

         
    volume_par *vpar;
    char vol_file[] = "testing_fodder/parameters/criteria.par";
    ck_assert_msg (file_exists(vol_file) == 1, "\n File %s does not exist\n", vol_file);    
    vpar = read_volume_par(vol_file);
    fail_if (vpar == NULL, "\n volume parameter file reading failed \n");
    
    
    
    control_par *cpar;
    char filename[] = "testing_fodder/parameters/ptv.par";
    ck_assert_msg (file_exists(filename) == 1, "\n File %s does not exist\n", filename);
    cpar = read_control_par(filename);
    fail_if (cpar == NULL, "\n control parameter file reading failed\n ");
    
    cpar->mm->lut = 0; // to start LUT initialization 
    cpar->num_cams = 1; // only one camera test

    mmlut test_mmlut[4];
    
    init_mmLUT (vpar, cpar, cal, test_mmlut);
    
    img_coord (X,Y,Z, cal, cpar->mm, i_cam, test_mmlut, &x, &y);
    
    ck_assert_msg(  fabs(x - 22.18131570) < EPS && 
                    fabs(y - 26.05917813)  < EPS,
     "Expected 22.18131570 26.05917813  but found %10.8f %10.8f\n", 
     x,y);
    
}
END_TEST
            






Suite* fb_suite(void) {
    Suite *s = suite_create ("Imgcoord");
 
    TCase *tc = tcase_create ("test_img_xy_mm_geo");
    tcase_add_test(tc, test_img_xy_mm_geo);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("test_imgcoord");
    tcase_add_test(tc, test_imgcoord);     
    suite_add_tcase (s, tc); 
      
    return s;
}

int main(void) {
    int number_failed;
    Suite *s = fb_suite ();
    SRunner *sr = srunner_create (s);
    //srunner_run_all (sr, CK_ENV);
    //srunner_run_all (sr, CK_SUBUNIT);
    srunner_run_all (sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int file_exists(char *filename){
    if( access(filename, F_OK ) != -1 ) {
        return 1;
    } else {
        printf("File %s does not exist\n",filename);
        return NULL;
    }
}
