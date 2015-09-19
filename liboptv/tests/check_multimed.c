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



#define EPS 1E-5

void print_Exterior(Exterior Ex_t);
int compare_exterior_diff(Exterior *e1, Exterior *e2);
int file_exists(char *filename);

START_TEST(test_init_mmLUT)
{
    double xmax, xmin, ymax, ymin, zmax, zmin;
    int i; 
    
    Calibration *cal;

    
    char ori_file[] = "testing_fodder/cal/cam3.tif.ori";
    char add_file[] = "testing_fodder/cal/cam3.tif.addpar";    
    cal = read_calibration(ori_file, add_file, NULL);
    
    if (cal == NULL) 
    {
        printf("\n cannot read calibration file \n");
    }
    else 
    {
        printf("\n Read calibration file successfully \n");
    }
    
         
    volume_par *vpar;
    
    char vol_file[] = "testing_fodder/parameters/criteri2a.par";
    
    ck_assert_msg (file_exists(vol_file) == 1, "\n File %s does not exist\n", vol_file);
    
    vpar = read_volume_par(vol_file);
    
    printf("%d\n",vpar);

    fail_if (vpar == NULL, "volume parameter file reading failed");
    
    
    
//     control_par *cpar;
//     char filename[] = "testing_fodder/parameters/ptv.par";
//     cpar = read_control_par(filename);
//     
//     printf("\n Read control parameters file successfully \n");
//     printf("cpar->mm->lut = %d \n",cpar->mm->lut);
//     
//      /* two default values which are not in the parameter file */
//     cpar->mm->lut = 0;
//     // cpar->mm->nlay = 1;
//         
// 
// 
// 
//      mmlut test_mmlut[4], correct_mmlut[4];  
//      
//      correct_mmlut[0].origin.x = 0.0;
//      correct_mmlut[0].origin.y = 0.0;
//      correct_mmlut[0].origin.z = -20.0;
//      correct_mmlut[0].nr = 13;
//      correct_mmlut[0].nz = 22;
//      correct_mmlut[0].rw = 2;
//              
//      init_mmLUT (vpar
//                , cpar
//                , cal
//                , test_mmlut);
                             
//      for (i=0; i<cpar->num_cams; i++){
//        ck_assert_msg( 
//                     fabs(test_mmlut[i].origin.x - correct_mmlut[0].origin.x) < EPS && 
//                     fabs(test_mmlut[i].origin.y - correct_mmlut[0].origin.y) < EPS && 
//                     fabs(test_mmlut[i].origin.z - correct_mmlut[0].origin.z)  < EPS &&
//                     test_mmlut[i].nr == correct_mmlut[i].nr &&
//                     test_mmlut[i].nz == correct_mmlut[i].nz &&
//                     test_mmlut[i].rw ==  correct_mmlut[i].rw &&
//                     fabs(test_mmlut[i].data[35] - 1.000) < EPS &&
//                     fabs(test_mmlut[i].data[82] - 1.016549) < EPS,
//          "\n Expected different correct_mmlut values but found: \n \
//          x,y,z = %8.6f %8.6f %8.6f \n nr,nz,rw = %d %d %d \n data = %8.6f %8.6f in camera %d \n", \
//          test_mmlut[i].origin.x, test_mmlut[i].origin.y, test_mmlut[i].origin.z, \
//          test_mmlut[i].nr, test_mmlut[i].nz, test_mmlut[i].rw, test_mmlut[i].data[35], test_mmlut[i].data[82], i);
//           
//        }    
}
END_TEST


START_TEST(test_back_trans_Point)
{
    double x = 100.0, y = 100.0, z =  0.0, X1,Y1,Z1;
        
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

     trans_Cam_Point(test_Ex, test_mm, test_G, x, y, z, &Ex_t, &X_t, &Y_t, &Z_t, \
     cross_p, cross_c);
    
     back_trans_Point(X_t, Y_t, Z_t, test_mm, test_G, cross_p, cross_c, &X1, &Y1, &Z1);
    
     ck_assert_msg( fabs(x - X1) < EPS && 
                    fabs(y - Y1) < EPS && 
                    fabs(z - Z1)  < EPS,
         "Expected %f, %f, %f  but found %f %f %f\n", x,y,z, X1, Y1, Z1);
      
    
}
END_TEST

// START_TEST(test_volumedimension)
// {
// 
//     /* input */
//     double xmax, xmin, ymax, ymin, zmax, zmin;
//     int i; 
//     
//     Calibration *cal;
//     char ori_file[] = "testing_fodder/cal/cam3.tif.ori";
//     char add_file[] = "testing_fodder/cal/cam3.tif.addpar";    
//     cal = read_calibration(ori_file, add_file, NULL);
//         
//     volume_par *vpar;
//     vpar = read_volume_par("testing_fodder/parameters/criteria_2.par");
//     
//     
//     control_par *cpar;
//     char filename[] = "testing_fodder/parameters/ptv_2.par";
//     cpar = read_control_par(filename);
//     cpar->mm->lut = 1;
//     cpar->mm->nlay = 1;
// 
//      
//      volumedimension (&xmax, &xmin, &ymax, &ymin, &zmax, &zmin, vpar, cpar, cal);
//      
//      
//     
//     
//      ck_assert_msg( fabs(xmax - 11.2017) < EPS && 
//                     fabs(xmin + 11.2017) < EPS && 
//                     fabs(ymax - 8.7392)  < EPS &&
//                     fabs(ymin + 8.7392)  < EPS &&
//                     fabs(zmax - 20.0000)  < EPS &&
//                     fabs(zmin + 20.0000)  < EPS,
//          "\n Expected 11.2017 -11.2017 8.7392 -8.7392 20.000 -20.000 \n  \
//          but found %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f \n", xmax, xmin, ymax, ymin, zmax, zmin);
//                
//     
// }
// END_TEST
// 
// 
// 



// START_TEST(test_get_mmf_mmLUT)
// {
// 
//     /* input */
//     double xmax, xmin, ymax, ymin, zmax, zmin;
//     int i, i_cam; 
//     
//     double mmf;
//     
//     Calibration *cal;
// 
//     
//     char ori_file[] = "testing_fodder/cal/cam3.tif.ori";
//     char add_file[] = "testing_fodder/cal/cam3.tif.addpar";    
//     cal = read_calibration(ori_file, add_file, NULL);
//     
//         
//     volume_par *vpar;
//     vpar = read_volume_par("testing_fodder/parameters/criteria_2.par");
//     
//     
//     control_par *cpar;
//     char filename[] = "testing_fodder/parameters/ptv_2.par";
//     cpar = read_control_par(filename);
//     
//     /* two default values which are not in the parameter file */
//     cpar->mm->lut = 0;  // to get mmLUT initialized
//     cpar->mm->nlay = 1;        
// 
//      mmlut test_mmlut[4], correct_mmlut[4];  
//      
//      correct_mmlut[0].origin.x = 0.0;
//      correct_mmlut[0].origin.y = 0.0;
//      correct_mmlut[0].origin.z = -70.0;
//      correct_mmlut[0].nr = 9;
//      correct_mmlut[0].nz = 47;
//      correct_mmlut[0].rw = 2;
//              
//      init_mmLUT (vpar
//                , cpar
//                , cal
//                , test_mmlut);
//                
//               
//      for (i=0; i<cpar->num_cams; i++){
//        ck_assert_msg( 
//                     fabs(test_mmlut[i].origin.x - correct_mmlut[0].origin.x) < EPS && 
//                     fabs(test_mmlut[i].origin.y - correct_mmlut[0].origin.y) < EPS && 
//                     fabs(test_mmlut[i].origin.z - correct_mmlut[0].origin.z)  < EPS &&
//                     test_mmlut[i].nr == correct_mmlut[i].nr &&
//                     test_mmlut[i].nz == correct_mmlut[i].nz &&
//                     test_mmlut[i].rw ==  correct_mmlut[i].rw &&
//                     fabs(test_mmlut[i].data[35] - 1.000000) < EPS &&
//                     fabs(test_mmlut[i].data[82] - 1.003130) < EPS,
//          "\n Expected different correct_mmlut values \n  \
//          but found %4.3f %4.3f %4.3f %d %d %d %4.3f %4.3f in camera %d\n", \
//          test_mmlut[i].origin.x, test_mmlut[i].origin.y, test_mmlut[i].origin.z, \
//          test_mmlut[i].nr, test_mmlut[i].nz, test_mmlut[i].rw,test_mmlut[i].data[35], \
//          test_mmlut[i].data[82], i);
//           
//         }
//         
//                
//          
//          for (i_cam = 0; i_cam < cpar->num_cams; i_cam++){          
//          
//          	mmf = get_mmf_from_mmLUT (i_cam, 1.0, 1.0, 1.0, (mmlut *) test_mmlut);
//         
//         	ck_assert_msg( 
//                     fabs(mmf - 1.002236) < EPS,
//          	"\n Expected mmf  1.002236 but found %8.6f in camera %d\n", \
//          	mmf, i_cam);
//          }
// }
// END_TEST
// 
// 
// 
// START_TEST(test_multimed_nlay)
// {
// 
//     /* input */
//     double xmax, xmin, ymax, ymin, zmax, zmin;
//     int i, i_cam; 
//         
//     Calibration *cal;
// 
//     
//     char ori_file[] = "testing_fodder/cal/cam3.tif.ori";
//     char add_file[] = "testing_fodder/cal/cam3.tif.addpar";    
//     cal = read_calibration(ori_file, add_file, NULL);
//     
//         
//     volume_par *vpar;
//     vpar = read_volume_par("testing_fodder/parameters/criteria_2.par");
//     
//     
//     control_par *cpar;
//     char filename[] = "testing_fodder/parameters/ptv_2.par";
//     cpar = read_control_par(filename);
//     /* two default values which are not in the parameter file */
//     cpar->mm->lut = 0;
//     cpar->mm->nlay = 1;
//             
// 
//      mmlut test_mmlut[4], correct_mmlut[4];  
//      
//      correct_mmlut[0].origin.x = 0.0;
//      correct_mmlut[0].origin.y = 0.0;
//      correct_mmlut[0].origin.z = -70.0;
//      correct_mmlut[0].nr = 9;
//      correct_mmlut[0].nz = 47;
//      correct_mmlut[0].rw = 2;
//              
//      init_mmLUT (vpar
//                , cpar
//                , cal
//                , test_mmlut);
//      
//                
//      for (i=0; i<cpar->num_cams; i++){
//        ck_assert_msg( 
//                     fabs(test_mmlut[i].origin.x - correct_mmlut[0].origin.x) < EPS && 
//                     fabs(test_mmlut[i].origin.y - correct_mmlut[0].origin.y) < EPS && 
//                     fabs(test_mmlut[i].origin.z - correct_mmlut[0].origin.z)  < EPS &&
//                     test_mmlut[i].nr == correct_mmlut[i].nr &&
//                     test_mmlut[i].nz == correct_mmlut[i].nz &&
//                     test_mmlut[i].rw ==  correct_mmlut[i].rw &&
//                     fabs(test_mmlut[i].data[35] - 1.000000) < EPS &&
//                     fabs(test_mmlut[i].data[82] - 1.003130) < EPS,
//          "\n Expected different correct_mmlut values \n  \
//          but found %4.3f %4.3f %4.3f %d %d %d %4.3 %4.3f in camera %d\n", \
//          test_mmlut[i].origin.x, test_mmlut[i].origin.y, test_mmlut[i].origin.z, \
//          test_mmlut[i].nr, test_mmlut[i].nz, test_mmlut[i].rw, test_mmlut[i].data[35], \
//          test_mmlut[i].data[82], i);
//           
//         }
//                                 
//      double X,Y,Z;
//      X = Y = Z = 1.23;
//      double correct_Xq,correct_Yq, Xq, Yq;
//      correct_Xq = correct_Yq = 1.2334;   
//      
//      i_cam = 0;
//                  
//      multimed_nlay ( &cal[0].ext_par
//                    , cpar->mm
//                    , X
//                    , Y
//                    , Z
//                    , &Xq
//                    , &Yq
//                    , i_cam 
//                    , (mmlut *) test_mmlut);
//         
//     for (i=0; i<cpar->num_cams; i++){
//        ck_assert_msg( 
//                     fabs(Xq - correct_Xq) < EPS && 
//                     fabs(Yq - correct_Yq) < EPS,
//          "\n Expected different correct_Xq, Yq values \n  \
//          but found %6.4f %6.4f \n", Xq, Yq);
//     }
// }
// END_TEST
// 
START_TEST(test_trans_Cam_Point)
{
    /* input */
    double x = 100.0;
    double y = 100.0;
    double z =  0.0;        
        
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
    double X_t, Y_t, Z_t;
    double cross_p[3], cross_c[3]; 

     trans_Cam_Point(test_Ex, test_mm, test_G, x, y, z, &Ex_t, &X_t, &Y_t, &Z_t, \
     cross_p, cross_c);
    
    
     ck_assert_msg( fabs(X_t - 141.421356) < EPS && 
                    fabs(Y_t - 0.0) < EPS && 
                    fabs(Z_t + 50.000000)  < EPS,
         "Expected 141.421356 0.000000 -50.000000  but found %f %f %f\n", X_t, Y_t, Z_t);
      
      
         
     ck_assert_msg( fabs(cross_p[0] - 100.000099) < EPS && 
                    fabs(cross_p[1] - 100.000010) < EPS && 
                    fabs(cross_p[2] - 50.00000)  < EPS,
         "Expected 100.0 100.0 50.000 but found %f %f %f\n", \
         cross_p[0],cross_p[1],cross_p[2]);
         
    
    
      ck_assert_msg(fabs(cross_c[0] + 0.0) < EPS && 
                    fabs(cross_c[1] + 0.0) < EPS && 
                    fabs(cross_c[2] - 55.0)  < EPS,
         "Expected 0.000000 0.000000 55.000000 but found %f %f %f\n", \
         cross_c[0],cross_c[1],cross_c[2]);
        
      /* print if necessary 
        print_Exterior(Ex_t);
        print_Exterior(correct_Ex_t);
      */
      
      
      fail_unless(compare_exterior_diff(&correct_Ex_t, &Ex_t));
      
    
}
END_TEST




Suite* fb_suite(void) {
    Suite *s = suite_create ("Multimedia model");
 
    TCase *tc = tcase_create ("Test init_mmLUT");
    tcase_add_test(tc, test_init_mmLUT);
    suite_add_tcase (s, tc);
    
//     tc = tcase_create ("Test trans_Cam_Point");
//     tcase_add_test(tc, test_trans_Cam_Point);
//     suite_add_tcase (s, tc);

//     tc = tcase_create ("Test back_trans_Point");
//     tcase_add_test(tc, test_back_trans_Point);
//     suite_add_tcase (s, tc);    
    
//   tc = tcase_create ("Test multimed_nlay");
//   tcase_add_test(tc, test_multimed_nlay);
//   suite_add_tcase (s, tc);    

//   tc = tcase_create ("Test get_mmf_mmLUT");
//   tcase_add_test(tc, test_get_mmf_mmLUT);
//   suite_add_tcase (s, tc);    

//   tc = tcase_create ("Test test_volumedimension");
//   tcase_add_test(tc, test_volumedimension);
//   suite_add_tcase (s, tc);    
          
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
        return NULL;
    }
}

