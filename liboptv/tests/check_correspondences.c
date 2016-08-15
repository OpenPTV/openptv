/*  Unit tests for functions related to finding calibration parameters. Uses
    the Check framework: http://check.sourceforge.net/

    To run it, type "make verify" when in the src/build/
    after installing the library.

    If that doesn't run the tests, use the Check tutorial:
    http://check.sourceforge.net/doc/check_html/check_3.html
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include "orientation.h"
#include "calibration.h"
#include "vec_utils.h"
#include "parameters.h"
#include "imgcoord.h"
#include "sortgrid.h"
#include "trafo.h"
#include "correspondences.h"

#define RO 200./M_PI


START_TEST(test_qs_target_y)
{
    target test_pix[] = {
        {0, 0.0, -0.2, 5, 1, 2, 10, -999},
        {6, 0.2, 0.0, 10, 8, 1, 20, -999},
        {3, 0.2, 0.8, 10, 3, 3, 30, -999},
        {4, 0.4, -1.1, 10, 3, 3, 40, -999},
        {1, 0.7, -0.1, 10, 3, 3, 50, -999},
        {7, 1.2, 0.3, 10, 3, 3, 60, -999},
        {5, 10.4, 0.1, 10, 3, 3, 70, -999}
    };

    /* sorting test_pix vertically by 'y' */
    qs_target_y (test_pix, 0, 6);

    /* first point should be -1.1 and the last 0.8 */
    fail_unless (fabs(test_pix[0].y + 1.1) < 1E-6);
    fail_unless (fabs(test_pix[1].y + 0.2) < 1E-6);
    fail_unless (fabs(test_pix[6].y - 0.8) < 1E-6);
}
END_TEST




Suite* orient_suite(void) {
    Suite *s = suite_create ("Testing correspondences");

    TCase *tc = tcase_create ("quicksort target y");
    tcase_add_test(tc, test_qs_target_y);
    suite_add_tcase (s, tc);

//     tc = tcase_create ("Point position");
//     tcase_add_test(tc, test_point_position);
//     suite_add_tcase (s, tc);
// 
//     tc = tcase_create ("Convergence measures");
//     tcase_add_test(tc, test_convergence_measure);
//     suite_add_tcase (s, tc);
//     
//     tc = tcase_create ("Raw orientation");
//     tcase_add_test(tc, test_raw_orient);
//     suite_add_tcase (s, tc);
// 
//     tc = tcase_create ("Orientation");
//     tcase_add_test(tc, test_orient);
//     suite_add_tcase (s, tc);

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
