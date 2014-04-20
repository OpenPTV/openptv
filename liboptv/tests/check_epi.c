/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "epi.h"

#define EPS 1E-4



START_TEST(test_epi_mm_2D)
{
        double x, y, z, xp, yp, zp;
        double pos[3], v[3];
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};
    
    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 1.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 0., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	1, 
    	1.0, 
    	{1.49, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.33,
    	1};
    	
    volume_par test_vpar = {
        {-250., 250.}, {-100., -100.}, {100., 100.}, 0.01, 0.3, 0.3, 0.01, 1.0, 33
        };
    /* trivial case */
     x = 0.0; 
     y = 0.0;
    
    epi_mm_2D (x, y, &test_cal, test_mm, &test_vpar, &xp, &yp, &zp);\

    
    ck_assert_msg( fabs(xp - 0.0) < EPS && 
                    fabs(yp - 0.0) < EPS && 
                    fabs(zp - 0.0)  < EPS,
         "\n Expected 0.0 0.0 0.0 \n  \
         but found %6.4f %6.4f %6.4f \n", xp, yp, zp);
    
}
END_TEST


START_TEST(test_find_candidate)
{

 
}
END_TEST

START_TEST(test_epi_mm)
{

      
    
}
END_TEST

            


Suite* fb_suite(void) {
    Suite *s = suite_create ("epi");
    TCase *tc = tcase_create ("epi_test");
     tcase_add_test(tc, test_epi_mm);
    tcase_add_test(tc, test_epi_mm_2D);    
    tcase_add_test(tc, test_find_candidate);
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

