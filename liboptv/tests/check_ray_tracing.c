/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"
#include "ray_tracing.h"

#define EPS 1E-5


START_TEST(test_ray_tracing)
{
    /* input */
    double x = 100.0;
    double y = 100.0;        
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.2, -0.3}, 
        {0.2, 1.0, 0.0},
        {-0.3, 0.0, 1.0}}};
    
    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 50.0};
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
    
    double X[3],a[3]; 


    
/*    
      ray_tracing (x, y, test_Ex, test_I, test_G, test_mm, \
      &X, &Y, &Z, \
      &a, &b, &c);
*/      
      ray_tracing (x, y, &test_cal, test_mm, (double *)X, (double *)a);

     
    
    
     ck_assert_msg( fabs(X[0] - 53.855021) < EPS && 
                    fabs(X[1] - 43.084017) < EPS && 
                    fabs(X[2] - test_G.vec_z)  < EPS,
         "Expected 110.393483, 88.314786, test_G.vec_z but found %f %f %f\n", X[0],X[1],X[2]);
      
      
         
     ck_assert_msg( fabs(a[0] - 0.387973) < EPS && 
                    fabs(a[1] - 0.310378) < EPS && 
                    fabs(a[2] + 0.867838)  < EPS,
         "Expected 0.387973, 0.310378,-0.867838 but found %g %g %g\n", a[0],a[1],a[2]);
         
}	
END_TEST
         
 START_TEST(test_trivial_ray_tracing)
{            
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};
    
    Interior test_I = {0.0, 0.0, 25.0};
    Glass test_G = {0.0, 0.0, 50.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 0., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	3, 
    	1.0, 
    	{1.0, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.0,
    	1};
    
    /* output */
    
    double X[3],a[3]; 

    	
    ray_tracing (0.0, 0.0, &test_cal, test_mm, (double *)X, (double *)a);

     
    
    
     ck_assert_msg( fabs(X[0] - 0.0) < EPS && 
                    fabs(X[1] - 0.0) < EPS && 
                    fabs(X[2] - test_G.vec_z)  < EPS,
         "Expected X = 0.0, 0.0, test_G.vec_z but found %g %g %g\n", X[0],X[1],X[2]);
      
      
         
     ck_assert_msg( fabs(a[0] - 0.0) < EPS && 
                    fabs(a[1] - 0.0) < EPS && 
                    fabs(a[2] + 1.0)  < EPS,
         "Expected a = 0.,0.,-1. but found %g %g %g\n", a[0],a[1],a[2]);

  
  
     ray_tracing (1.0, 10.0, &test_cal, test_mm, (double *)X, (double *)a);
    
    
     ck_assert_msg( fabs(X[0] - 2.0) < EPS && 
                    fabs(X[1] - 20.0) < EPS && 
                    fabs(X[2] - test_G.vec_z)  < EPS,
         "Expected X = 1.0, 1.0, test_G.vec_z but found %g %g %g\n", X[0],X[1],X[2]);
      
      
         
     ck_assert_msg( fabs(a[0] - 0.037113) < EPS && 
                    fabs(a[1] - 0.371135) < EPS && 
                    fabs(a[2] + 0.927837)  < EPS,
         "Expected a = 0.037113, 0.371135, -0.927837 but found %f %f %f\n", a[0],a[1],a[2]);
    
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("ray_tracing");
 
    TCase *tc = tcase_create ("ray_tracing_test");
    tcase_add_test(tc, test_ray_tracing);
    tcase_add_test(tc, test_trivial_ray_tracing);
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

