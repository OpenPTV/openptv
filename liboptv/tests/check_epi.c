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
#include "epi.h"

#define EPS 1E-4



START_TEST(test_epi_mm)
{

      
    
}
END_TEST


START_TEST(test_epi_mm_2D)
{
 
      
    
}
END_TEST


START_TEST(test_find_candidate_plus_msg)
{

 
}
END_TEST



START_TEST(test_find_candidate_plus)
{


    
    
}
END_TEST
            


Suite* fb_suite(void) {
    Suite *s = suite_create ("epi");
 
    TCase *tc = tcase_create ("epi_test");

    tcase_add_test(tc, test_epi_mm);
    tcase_add_test(tc, test_epi_mm_2D);    
    tcase_add_test(tc, test_find_candidate_plus_msg);
    tcase_add_test(tc, test_find_candidate_plus);     
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

