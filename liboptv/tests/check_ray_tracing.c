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
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	1, 
    	1.0, 
    	{1.33, 0.0, 0.0}, 
    	{6.0, 0.0, 0.0},
    	1.46,
    	1};
    
    /* output */
    double X[3],v[3]; 


    
    ray_tracing (x, y, &test_cal, test_mm, (double *)X, (double *)v);

     
   
    ck_assert_msg(  fabs(X[0] + 35.703066)   < EPS && 
                    fabs(X[1] - 56.107435)  < EPS && 
                    fabs(X[2] - 124.999998) < EPS,
         "Expected -35.703066 56.107435 124.999998 but found %f %f %f\n", X[0],X[1],X[2]);
      
      
        
     ck_assert_msg( fabs(v[0] + 0.235876) < EPS && 
                    fabs(v[1] + 0.019007) < EPS && 
                    fabs(v[2] + 0.971597)  < EPS,
         "Expected -0.235876 -0.019007 -0.971597 but found %f %f %f\n", v[0],v[1],v[2]);
           
    
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

