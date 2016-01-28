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
    Glass test_G = {0.0001, 0.00001, 1.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	3, 
    	1.0, 
    	{1.49, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.33};
    
    /* output */
    
    double X[3],a[3]; 


    
/*    
      ray_tracing (x, y, test_Ex, test_I, test_G, test_mm, \
      &X, &Y, &Z, \
      &a, &b, &c);
*/      
      ray_tracing (x, y, &test_cal, test_mm, (double *)X, (double *)a);

     
    
    
     ck_assert_msg( fabs(X[0] - 110.406944) < EPS && 
                    fabs(X[1] - 88.325788) < EPS && 
                    fabs(X[2] - 0.988076)  < EPS,
         "Expected 110.406944, 88.325788, 0.988076 but found %f %f %f\n", X);
      
      
         
     ck_assert_msg( fabs(a[0] - 0.387960) < EPS && 
                    fabs(a[1] - 0.310405) < EPS && 
                    fabs(a[2] + 0.867834)  < EPS,
         "Expected 0.387960,0.310405,-0.867834 but found %f %f %f\n", a);
  
    
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("ray_tracing");
 
    TCase *tc = tcase_create ("ray_tracing_test");
    tcase_add_test(tc, test_ray_tracing);
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

