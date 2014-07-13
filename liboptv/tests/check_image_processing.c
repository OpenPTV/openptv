/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "epi.h"
#include "image_processing.h"

#define EPS 1E-4



START_TEST(test_filter_3)
{
        double x, y, z, xp, yp, zp;
        double pos[3], v[3];
        
    
            
    ck_assert_msg(  fabs(0.0 - 0.0) < EPS && 
                    fabs(0.0 - 0.0) < EPS && 
                    fabs(0.0 - 0.0)  < EPS,
         "\n Expected 0.0 0.0 0.0 \n  \
         but found %6.4f %6.4f %6.4f \n", xp, yp, zp);
    
}
END_TEST

START_TEST(test_filter_3)
{
        double x, y, z, xp, yp, zp;
        double pos[3], v[3];
        
    
            
    ck_assert_msg(  fabs(0.0 - 0.0) < EPS && 
                    fabs(0.0 - 0.0) < EPS && 
                    fabs(0.0 - 0.0)  < EPS,
         "\n Expected 0.0 0.0 0.0 \n  \
         but found %6.4f %6.4f %6.4f \n", xp, yp, zp);
    
}
END_TEST



Suite* fb_suite(void) {
    Suite *s = suite_create ("image_processing");
    TCase *tc = tcase_create ("image_processing_test");
    tcase_add_test(tc, test_filter_3);
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

