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

#define EPS 1E-5

void print_Exterior(Exterior Ex_t);


START_TEST(test_trans_Cam_Point_back)
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
        0.0, 0.0, 94.0,
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

     printf("Entered test_trans_Cam_Point_back \n");
     trans_Cam_Point_back(test_Ex, test_mm, test_G, x, y, z, &Ex_t, &X_t, &Y_t, &Z_t, \
     cross_p, cross_c);
    
    
     ck_assert_msg( fabs(X_t - 141.429134) < EPS && 
                    fabs(Y_t - 0.0) < EPS && 
                    fabs(Z_t + 5.991000)  < EPS,
         "Expected 141.429134 0.000000 -5.991000  but found %f %f %f\n", X_t, Y_t, Z_t);
      
      
         
     ck_assert_msg( fabs(cross_p[0] - 100.000099) < EPS && 
                    fabs(cross_p[1] - 100.000010) < EPS && 
                    fabs(cross_p[2] - 0.991000)  < EPS,
         "Expected 100.0 100.0 0.991 but found %f %f %f\n", cross_p[0],cross_p[1],cross_p[2]);
         
    
    
      ck_assert_msg(fabs(cross_c[0] + 0.009400) < EPS && 
                    fabs(cross_c[1] + 0.000940) < EPS && 
                    fabs(cross_c[2] - 6.000001)  < EPS,
         "Expected -0.009400 -0.000940 6.000001 but found %f %f %f\n", cross_c[0],cross_c[1],cross_c[2]);
        
      
      print_Exterior(correct_Ex_t);
      print_Exterior(Ex_t);
      
      ck_assert_msg(compare_exterior(&correct_Ex_t, &Ex_t) == 1, 
         "Expected different Exterior parameters, see above \n");
      
    
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("multimed");
 
    TCase *tc = tcase_create ("multimed_test");
    tcase_add_test(tc, test_trans_Cam_Point_back);
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

