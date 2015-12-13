/*  Unit tests for functions related to finding calibration parameters. Uses 
    the Check framework: http://check.sourceforge.net/
    
    To run it, type "make check" when in the top C directory, src_c/
    If that doesn't run the tests, use the Check tutorial:
    http://check.sourceforge.net/doc/check_html/check_3.html
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "orientation.h"
#include "vec_utils.h"
#include "parameters.h"

START_TEST(test_ray_distance)
{
    /* Generate simply-oriented skew rays with a known distance and get that
       distance from the distance finding routine. */
    
    vec3d pos1 = {0., 0., 0.}, dir1 = {1., 0., 0.};
    vec3d pos2 = {0., 0., 1.}, dir2 = {0., 1., 0.};
    
    /* cases: skew, intersecting, parallel */
    fail_unless(ray_distance(pos1, dir1, pos2, dir2) == 1.);
    fail_unless(ray_distance(pos1, dir1, pos1, dir2) == 0.);
    fail_unless(ray_distance(pos1, dir1, pos2, dir1) == 1.);
}
END_TEST


Suite* orient_suite(void) {
    Suite *s = suite_create ("Finding calibration parameters");

    TCase *tc = tcase_create ("Ray distance");
    tcase_add_test(tc, test_ray_distance);
    suite_add_tcase (s, tc);

    return s;
}

int main(void) {
    int number_failed;
    Suite *s = orient_suite();
    SRunner *sr = srunner_create (s);
    srunner_run_all (sr, CK_ENV);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

