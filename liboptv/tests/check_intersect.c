/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"

#define EPS 1E-5


START_TEST(test_empty)
{

    double n[3];

    // test simple cross-product normalized to unity

    double a[] = {1.0, 0.0, 0.0};
    double b[] = {0.0, 2.0, 0.0};


    ck_assert_msg( fabs(a[0] - 1.0) < EPS && 
                   fabs(a[1] - 0.0) < EPS && 
                   fabs(a[2] - 0.0)  < EPS,
             "Was expecting a to be 0., 0., 0. but found %f %f %f\n", a[0],a[1],a[2]);


}
END_TEST



Suite* fb_suite(void) {
    Suite *s = suite_create ("intersect");
 
    TCase *tc = tcase_create ("intersect test");
    tcase_add_test(tc, test_empty);
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

