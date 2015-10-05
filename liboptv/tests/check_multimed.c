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
#include <unistd.h>
#include "vec_utils.h"



#define EPS 1E-6

void print_Exterior(Exterior Ex_t);
int compare_exterior_diff(Exterior *e1, Exterior *e2);
int file_exists(char *filename);

START_TEST(test_init_mmLUT)
{
    double xmax, xmin, ymax, ymin, zmax, zmin;
    int i; 
    
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

    /* lut value is not in the parameter file */
    cpar->mm->lut = 0;
    mmlut test_mmlut[4], correct_mmlut[4]; 
     
    vec_set(correct_mmlut[0].origin, 0.0, 0.0, -250.00001105);

    correct_mmlut[0].nr = 130;
    correct_mmlut[0].nz = 177;
    correct_mmlut[0].rw = 2;
     
    /* run init_mmLUT for one camera only */
    i = 0;
    cpar->num_cams = 1;
             
    init_mmlut (vpar, cpar, cal);
    ck_assert_msg( 
        fabs(cal->mmlut.origin[0] - correct_mmlut[0].origin[0]) < EPS && 
        fabs(cal->mmlut.origin[1]- correct_mmlut[0].origin[1]) < EPS && 
        fabs(cal->mmlut.origin[2] - correct_mmlut[0].origin[2])  < EPS &&
        cal->mmlut.nr == correct_mmlut[i].nr &&
        cal->mmlut.nz == correct_mmlut[i].nz &&
        cal->mmlut.rw ==  correct_mmlut[i].rw &&
        fabs(cal->mmlut.data[0] - 1.11089711) < EPS &&
        fabs(cal->mmlut.data[200] - 1.09709147) < EPS,
        "\n Expected different correct_mmlut values but found: \n \
        x,y,z = %10.8f %10.8f %10.8f \n nr,nz,rw = %d %d %d \n data = %10.8f %10.8f \
         in camera %d \n", 
        cal->mmlut.origin[0], cal->mmlut.origin[1], cal->mmlut.origin[2], \
        cal->mmlut.nr, cal->mmlut.nz, cal->mmlut.rw, cal->mmlut.data[0], 
        cal->mmlut.data[200], i
    );
}
END_TEST


START_TEST(test_back_trans_Point)
{
    vec3d pos1, pos_t, pos = {100.0, 100.0, 0.0};
    
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.2, -0.3}, 
        {0.2, 1.0, 0.0},
        {-0.3, 0.0, 1.0}}};
        
    Exterior correct_Ex_t = {
        0.0, 0.0, 99.0,
        -0.0, 0.0, 0.0, 
        {{-0.0, -0.0, -0.0}, 
        {-0.0, 0.0, -0.0},
        {0.0, -0.0, -0.0}}};
    
    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0001, 0.00001, 1.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	1, 
    	1.0, 
    	{1.49, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.33,
    	1};
    
    Exterior Ex_t; 
    double X_t, Y_t, Z_t;
    double cross_p[3], cross_c[3]; 

     trans_Cam_Point(test_Ex, test_mm, test_G, pos, &Ex_t, pos_t, \
     cross_p, cross_c);
    
     back_trans_Point(pos_t, test_mm, test_G, cross_p, cross_c, pos1);
    
     ck_assert_msg( fabs(pos1[0] - pos[0]) < EPS && 
                    fabs(pos1[1] - pos[1]) < EPS && 
                    fabs(pos1[2] - pos[2])  < EPS,
         "Expected %f, %f, %f  but found %f %f %f\n", pos[0],pos[1],pos[2], pos1[0], 
            pos1[1], pos1[2]);
      
    
}
END_TEST

START_TEST(test_volumedimension)
{

    double xmax, xmin, ymax, ymin, zmax, zmin;
    int i; 
    
    Calibration *tmp, cal[2];
        
    char ori_file[] = "testing_fodder/cal/cam1.tif.ori";
    char add_file[] = "testing_fodder/cal/cam1.tif.addpar";
    
    ck_assert_msg (file_exists(ori_file) == 1, "\n File %s does not exist\n", ori_file);
    ck_assert_msg (file_exists(add_file) == 1, "\n File %s does not exist\n", add_file);
    tmp = read_calibration(ori_file, add_file, NULL);    
    fail_if (tmp == NULL, "\n ORI or ADDPAR file reading failed \n");
    cal[0] = *tmp;
    
    
    char ori_file2[] = "testing_fodder/cal/cam2.tif.ori";
    char add_file2[] = "testing_fodder/cal/cam2.tif.addpar";
    
    ck_assert_msg (file_exists(ori_file2) == 1, "\n File %s does not exist\n", ori_file2);
    ck_assert_msg (file_exists(add_file2) == 1, "\n File %s does not exist\n", add_file2);
    tmp = read_calibration(ori_file, add_file2, NULL);    
    fail_if (tmp == NULL, "\n ORI or ADDPAR file reading failed \n");
    cal[1] = *tmp;
    
         
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

    cpar->mm->lut = 1;
    cpar->mm->nlay = 1;
    cpar->num_cams = 2;

    volumedimension (&xmax, &xmin, &ymax, &ymin, &zmax, &zmin, vpar, cpar, cal);
    
    ck_assert_msg( fabs(xmax - 73.02053752) < EPS && 
        fabs(xmin + 46.80667189) < EPS && 
        fabs(ymax - 51.04924925)  < EPS &&
        fabs(ymin + 62.91848990)  < EPS &&
        fabs(zmax - 100.0000)  < EPS &&
        fabs(zmin + 100.0000)  < EPS,
        "\n Expected 73.02053752 -46.80667189 51.04924925 -62.91848990 100.0000 -100.0000 \n  \
        but found %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f \n", xmax, xmin, ymax, 
        ymin, zmax, zmin
    );           
}
END_TEST


START_TEST(test_get_mmf_mmLUT)
{
    double mmf; 
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

    /* lut value is no in the parameter file */
    cpar->mm->lut = 0;

    mmlut correct_mmlut[4]; 
     
    vec_set(correct_mmlut[0].origin, 0.0, 0.0, -250.00001105);
    correct_mmlut[0].nr = 130;
    correct_mmlut[0].nz = 177;
    correct_mmlut[0].rw = 2;
    
    init_mmlut (vpar, cpar, cal);
     
    ck_assert_msg( 
        fabs(cal->mmlut.origin[0] - correct_mmlut[0].origin[0]) < EPS && 
        fabs(cal->mmlut.origin[1] - correct_mmlut[0].origin[1]) < EPS && 
        fabs(cal->mmlut.origin[2] - correct_mmlut[0].origin[2])  < EPS &&
        cal->mmlut.nr == correct_mmlut[0].nr &&
        cal->mmlut.nz == correct_mmlut[0].nz &&
        cal->mmlut.rw ==  correct_mmlut[0].rw &&
        fabs(cal->mmlut.data[0] - 1.11089711) < EPS &&
        fabs(cal->mmlut.data[200] - 1.09709147) < EPS,
        "\n Expected different correct_mmlut values but found: \n \
        x,y,z = %10.8f %10.8f %10.8f \n nr,nz,rw = %d %d %d \n data = %10.8f %10.8f\n",
        cal->mmlut.origin[0], cal->mmlut.origin[1], cal->mmlut.origin[2], \
        cal->mmlut.nr, cal->mmlut.nz, cal->mmlut.rw, cal->mmlut.data[0], 
        cal->mmlut.data[200]); 
    
    vec3d pos = {1.0, 1.0, 1.0}; 
    mmf = get_mmf_from_mmlut (cal, pos);
    ck_assert_msg(fabs(mmf - 1.00363015) < EPS,
        "\n Expected mmf  1.00363015 but found %10.8f\n", mmf);
}
END_TEST

 
START_TEST(test_multimed_nlay)
{
    double xmax, xmin, ymax, ymin, zmax, zmin;
    int i, i_cam; 
        
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

    mmlut correct_mmlut[4];  
     
    vec_set(correct_mmlut[0].origin, 0.0, 0.0, -250.00003540);
    
    correct_mmlut[0].nr = 114;
    correct_mmlut[0].nz = 177;
    correct_mmlut[0].rw = 2;
             
    init_mmlut (vpar, cpar, cal);
     
    vec3d pos = {1.23, 1.23, 1.23};
    double correct_Xq,correct_Yq, Xq, Yq;
    correct_Xq = 0.85954957; 
    correct_Yq = 0.86851375;   
     
    i_cam = 0;
                 
    multimed_nlay(cal, cpar->mm, pos, &Xq, &Yq);
        
    for (i=0; i < cpar->num_cams; i++){
       ck_assert_msg( 
         fabs(Xq - correct_Xq) < EPS && 
         fabs(Yq - correct_Yq) < EPS,
         "\n Expected different correct_Xq, Yq values \n  \
         but found %10.8f %10.8f \n", Xq, Yq);
    }
}
END_TEST


START_TEST(test_trans_Cam_Point)
{
    /* input */  
    vec3d pos = {100.0, 100.0, 0.0};     
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.2, -0.3}, 
        {0.2, 1.0, 0.0},
        {-0.3, 0.0, 1.0}}};
        
    Exterior correct_Ex_t = {
        0.0, 0.0, 50.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.2, -0.3}, 
        {0.2, 1.0, 0.0},
        {-0.3, 0.0, 1.0}}};
    
    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 50.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	1, 
    	1.0, 
    	{1.49, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.33,
    	1};
    
    /* output */
    Exterior Ex_t; 
    vec3d pos_t;
    double cross_p[3], cross_c[3]; 

    trans_Cam_Point(test_Ex, test_mm, test_G, pos, &Ex_t, pos_t, cross_p, cross_c);
    
    ck_assert_msg( fabs(pos_t[0] - 141.421356) < EPS && 
        fabs(pos_t[1] - 0.0) < EPS && 
        fabs(pos_t[2] + 50.000000)  < EPS,
        "Expected 141.421356 0.000000 -50.000000  but found %10.8f %10.8f %10.8f\n", 
        pos_t[0],pos_t[1],pos_t[2]);
         
    ck_assert_msg( fabs(cross_p[0] - 100.0) < EPS && 
        fabs(cross_p[1] - 100.0) < EPS && 
        fabs(cross_p[2] - 50.0)  < EPS,
        "Expected 100.0 100.0 50.000 but found %10.8f %10.8f %10.8f\n", \
        cross_p[0],cross_p[1],cross_p[2]);
    
    ck_assert_msg(fabs(cross_c[0] + 0.0) < EPS && 
        fabs(cross_c[1] + 0.0) < EPS && 
        fabs(cross_c[2] - 55.0)  < EPS,
        "Expected 0.000000 0.000000 55.000000 but found %10.8f %10.8f %10.8f\n", \
        cross_c[0],cross_c[1],cross_c[2]);
        
    fail_unless(compare_exterior_diff(&correct_Ex_t, &Ex_t));
}
END_TEST


START_TEST(test_move_along_ray)
{
    double glob_Z = 2;
    vec3d vertex = {1, 1, 1};
    vec3d direct = {1, 1, 1};
    vec3d correct = {2, 2, 2};
    vec3d res;
    
    move_along_ray(glob_Z, vertex, direct, res);
    printf("%g %g %g\n", res[0], res[1], res[2]);
    fail_unless(vec_cmp(res, correct));
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("Multimedia model");
 
    TCase *tc = tcase_create ("Test init_mmLUT");
    tcase_add_test(tc, test_init_mmLUT);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Test trans_Cam_Point");
    tcase_add_test(tc, test_trans_Cam_Point);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Test back_trans_Point");
    tcase_add_test(tc, test_back_trans_Point);
    suite_add_tcase (s, tc);    
    
    tc = tcase_create ("Test multimed_nlay");
    tcase_add_test(tc, test_multimed_nlay);
    suite_add_tcase (s, tc);    

    tc = tcase_create ("Test get_mmf_mmLUT");
    tcase_add_test(tc, test_get_mmf_mmLUT);
    suite_add_tcase (s, tc);    

    tc = tcase_create ("Test test_volumedimension");
    tcase_add_test(tc, test_volumedimension);
    suite_add_tcase (s, tc);    
          
    tc = tcase_create ("Test move_along_ray()");
    tcase_add_test(tc, test_move_along_ray);
    suite_add_tcase (s, tc);
    
    return s;
}

int main(void) {
    int number_failed;
    Suite *s = fb_suite ();
    SRunner *sr = srunner_create (s);
    srunner_run_all (sr, CK_ENV);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

void print_Exterior (Exterior Ex){
  int i;
  printf ("Exterior parameters \n");
  printf ("%11.4f %11.4f %11.4f\n    %10.7f  %10.7f  %10.7f\n\n",
	   Ex.x0, Ex.y0, Ex.z0, Ex.omega, Ex.phi, Ex.kappa);
  for (i=0; i<3; i++)  printf ("    %10.7f %10.7f %10.7f\n",
				Ex.dm[i][0], Ex.dm[i][1], Ex.dm[i][2]);
}

int compare_exterior_diff(Exterior *e1, Exterior *e2) {
    int row, col;
    
    for (row = 0; row < 3; row++)
        for (col = 0; col < 3; col++)
            if (fabs(e1->dm[row][col] - e2->dm[row][col]) > EPS)
                return 0;
    
    return ((fabs(e1->x0 - e2->x0) < EPS) && (fabs(e1->y0 - e2->y0) < EPS) \
    && (fabs(e1->z0 - e2->z0) < EPS) \
        && (fabs(e1->omega - e2->omega) < EPS) && (fabs(e1->phi - e2->phi) < EPS)\
        && (fabs(e1->kappa - e2->kappa) < EPS));
}


int file_exists(char *filename){
    if( access(filename, F_OK ) != -1 ) {
        return 1;
    } else {
        printf("File %s does not exist\n",filename);
        return 0;
    }
}

