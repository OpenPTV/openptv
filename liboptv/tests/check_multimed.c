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

#define EPS 1E-3

void print_Exterior(Exterior Ex_t);
int compare_exterior_diff(Exterior *e1, Exterior *e2);


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
    	3, 
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
    
    
     ck_assert_msg( fabs(X_t - 141.429134) < EPS && 
                    fabs(Y_t - 0.0) < EPS && 
                    fabs(Z_t + 0.989000)  < EPS,
         "Expected 141.429134 0.000000 -0.989000  but found %f %f %f\n", X_t, Y_t, Z_t);
      
      
         
     ck_assert_msg( fabs(cross_p[0] - 100.000099) < EPS && 
                    fabs(cross_p[1] - 100.000010) < EPS && 
                    fabs(cross_p[2] - 0.989000)  < EPS,
         "Expected 100.0 100.0 0.989 but found %f %f %f\n", \
         cross_p[0],cross_p[1],cross_p[2]);
         
    
    
      ck_assert_msg(fabs(cross_c[0] + 0.009400) < EPS && 
                    fabs(cross_c[1] + 0.000940) < EPS && 
                    fabs(cross_c[2] - 6.000001)  < EPS,
         "Expected -0.009400 -0.000940 6.000001 but found %f %f %f\n", \
         cross_c[0],cross_c[1],cross_c[2]);
        
      /* print_Exterior(Ex_t); */
      
      fail_unless(compare_exterior_diff(&correct_Ex_t, &Ex_t));
      
    
}
END_TEST


START_TEST(test_back_trans_Point)
{
    /* input */
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
    	3, 
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
    
     back_trans_Point(X_t, Y_t, Z_t, test_mm, test_G, cross_p, cross_c, &X1, &Y1, &Z1);
    
     ck_assert_msg( fabs(x - X1) < EPS && 
                    fabs(y - Y1) < EPS && 
                    fabs(z - Z1)  < EPS,
         "Expected %f, %f, %f  but found %f %f %f\n", x,y,z, X1, Y1, Z1);
      
    
}
END_TEST


START_TEST(test_volumedimension)
{

    /* input */
    double xmax, xmin, ymax, ymin, zmax, zmin;
    int i; 
    
    Calibration test_cal[4];
            
     /* input */
    double x = -7.713157;
    double y = 6.144260;        
        
    Exterior test_Ex = {
        128.011300, 69.300100, 572.731900,
        -0.121629, 0.242729, 0.005532, 
        {{0.970671, -0.005369, 0.240352}, 
        {-0.023671 ,  0.992758 ,  0.117773},
        {-0.239244 ,  -0.120008 ,  0.963515}}};
    
    Interior test_I = {0.0, 0.0, 70.0};
    Glass test_G = {0.000010, 0.000010, 125.000000};
    ap_52 test_addp = {0.0, 0.0, 0.0, 0.0, 0.0, 1.003025, -0.009194};
    
    /*
     Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    */ 
    
    mm_np test_mm = {
    	3, 
    	1.0, 
    	{1.33, 0.0, 0.0}, 
    	{6.0, 0.0, 0.0},
    	1.46,
    	1};
    
    
    
    control_par test_cpar;
    volume_par test_vpar; 
    
    
    /* additional initial values */
    
    test_cpar.imx = 1280; 
    test_cpar.imy = 1024;
    test_cpar.pix_x = 0.012;
    test_cpar.pix_y = 0.012;
    test_cpar.num_cams = 4;
    test_cpar.mm = &test_mm;
    
    
    /* test values for zmin,zmax,xmax,xmin */
    test_vpar.Zmin_lay[0] = -20.0;
    test_vpar.Zmax_lay[0] = 20.0;
    test_vpar.X_lay[1]    = 40.0;
    test_vpar.X_lay[0]    = -40.0;

    
     
    for (i=0; i<test_cpar.num_cams; i++){
     	test_cal[i].ext_par = test_Ex;
     	test_cal[i].int_par = test_I;
     	test_cal[i].glass_par = test_G;
     	test_cal[i].added_par =  test_addp;
     }


     
     volumedimension (&xmax, &xmin, &ymax, &ymin, &zmax, &zmin, \
     &test_vpar, &test_cpar, test_cal);
    
    
     ck_assert_msg( fabs(xmax - 57.892) < EPS && 
                    fabs(xmin + 73.420) < EPS && 
                    fabs(ymax - 54.053)  < EPS &&
                    fabs(ymin + 48.745)  < EPS &&
                    fabs(zmax - 20.00)  < EPS &&
                    fabs(zmin + 20.00)  < EPS,
         "\n Expected 57.892 -73.420 54.053 -48.745 20.000 -20.000 \n  \
         but found %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f \n", xmax, xmin, ymax, ymin, zmax, zmin);
      
    
}
END_TEST



START_TEST(test_init_mmLUT)
{

    /* input */
    double xmax, xmin, ymax, ymin, zmax, zmin;
    int i; 
    
    Calibration test_cal[4];
            
     /* input */
    double x = -7.713157;
    double y = 6.144260;        
        
    Exterior test_Ex = {
        128.011300, 69.300100, 572.731900,
        -0.121629, 0.242729, 0.005532, 
        {{0.970671, -0.005369, 0.240352}, 
        {-0.023671 ,  0.992758 ,  0.117773},
        {-0.239244 ,  -0.120008 ,  0.963515}}};
    
    Interior test_I = {0.0, 0.0, 70.0};
    Glass test_G = {0.000010, 0.000010, 125.000000};
    ap_52 test_addp = {0.0, 0.0, 0.0, 0.0, 0.0, 1.003025, -0.009194};
    
    /*
     Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    */ 
    
    mm_np test_mm = {
    	3, 
    	1.0, 
    	{1.33, 0.0, 0.0}, 
    	{6.0, 0.0, 0.0},
    	1.46,
    	1};
    
    
    
    control_par test_cpar;
    volume_par test_vpar; 
    
    mmlut test_mmlut[4], correct_mmlut[4];
    
    
    test_cpar.imx = 1280; 
    test_cpar.imy = 1024;
    test_cpar.pix_x = 0.012;
    test_cpar.pix_y = 0.012;
    test_cpar.num_cams = 4;
    test_cpar.mm = &test_mm;
    
    
    /* test values for zmin,zmax,xmax,xmin */
    test_vpar.Zmin_lay[0] = -20.0;
    test_vpar.Zmax_lay[0] = 20.0;
    test_vpar.X_lay[1]    = 40.0;
    test_vpar.X_lay[0]    = -40.0;

    
     
    for (i=0; i<test_cpar.num_cams; i++){
     	test_cal[i].ext_par = test_Ex;
     	test_cal[i].int_par = test_I;
     	test_cal[i].glass_par = test_G;
     	test_cal[i].added_par =  test_addp;
     }
     
     
     correct_mmlut[0].origin.x = 0.0;
     correct_mmlut[0].origin.y = 0.0;
     correct_mmlut[0].origin.z = -125.0;
     correct_mmlut[0].nr = 115;
     correct_mmlut[0].nz = 64;
     correct_mmlut[0].rw = 2;
     
     correct_mmlut[1].origin.x = 0.0;
     correct_mmlut[1].origin.y = 0.0;
     correct_mmlut[1].origin.z = -125.0;
     correct_mmlut[1].nr = 116;
     correct_mmlut[1].nz = 65;
     correct_mmlut[1].rw = 2;
 
     correct_mmlut[2].origin.x = 0.0;
     correct_mmlut[2].origin.y = 0.0;
     correct_mmlut[2].origin.z = -125.0;
     correct_mmlut[2].nr = 117;
     correct_mmlut[2].nz = 66;
     correct_mmlut[2].rw = 2;
     
     correct_mmlut[3].origin.x = 0.0;
     correct_mmlut[3].origin.y = 0.0;
     correct_mmlut[3].origin.z = -125.0;
     correct_mmlut[3].nr = 118;
     correct_mmlut[3].nz = 67;
     correct_mmlut[3].rw = 2;    
     
               
     init_mmLUT (
                 &test_vpar
               , &test_cpar
               , test_cal
               , &test_mmlut);
                   
    for (i=0; i<4; i++){
       ck_assert_msg( 
                    fabs(test_mmlut[i].origin.x - correct_mmlut[0].origin.x) < EPS && 
                    fabs(test_mmlut[i].origin.y - correct_mmlut[0].origin.y) < EPS && 
                    fabs(test_mmlut[i].origin.z - correct_mmlut[0].origin.z)  < EPS &&
                    test_mmlut[i].nr == correct_mmlut[i].nr &&
                    test_mmlut[i].nz == correct_mmlut[i].nz &&
                    test_mmlut[i].rw ==  correct_mmlut[i].rw,
         "\n Expected different correct_mmlut values \n  \
         but found %4.3f %4.3f %4.3f %d %d %d in camera %d\n", \
         test_mmlut[i].origin.x, test_mmlut[i].origin.y, test_mmlut[i].origin.z, \
         test_mmlut[i].nr, test_mmlut[i].nz, test_mmlut[i].rw, i);
    }
      
    
}
END_TEST



Suite* fb_suite(void) {
    Suite *s = suite_create ("multimed");
 
    TCase *tc = tcase_create ("multimed_test");
    tcase_add_test(tc, test_volumedimension);
    tcase_add_test(tc, test_init_mmLUT);
    tcase_add_test(tc, test_trans_Cam_Point);
    tcase_add_test(tc, test_back_trans_Point); 
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
