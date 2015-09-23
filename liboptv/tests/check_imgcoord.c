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

    vec3d pos = {50.0, 100.0, -10.0};
    double x,y ;
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
    
    cpar->num_cams = 1; // only one camera test
    
    init_mmlut (vpar, cpar, cal);
    
    img_xy_mm_geo (pos, cal, cpar->mm, &x, &y);
    
    ck_assert_msg(  fabs(x + 0.29374895) < EPS && 
                    fabs(y - 33.79886299)  < EPS,
     "Expected -0.29374895 33.79886299  but found %10.8f %10.8f\n", 
     x,y);      
    
}
END_TEST



START_TEST(test_imgcoord)
{
    vec3d pos = {50.0, 100.0, -10.0};
    double x,y ;
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
    
    
    cpar->num_cams = 1; // only one camera test
    
    init_mmlut (vpar, cpar, cal);
    
    img_coord (pos, cal, cpar->mm, &x, &y);
    
    ck_assert_msg(  fabs(x - 10.65513015) < EPS && 
                    fabs(y - 26.18672754)  < EPS,
     "Expected 10.65513015 26.18672754  but found %10.8f %10.8f\n", 
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
