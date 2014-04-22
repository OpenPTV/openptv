
/*  Unit-test suit for the C core of PyPTV. Uses the Check framework:
    http://check.sourceforge.net/
    
    To run it, type "make verify" when in the build directory.
    
    References:
    [1] Craig, J.J., Introduction to Robotics, 2nd ed.
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "calibration.h"

/* Generate a calibration object with example values matching those in the
   files read by test_read_ori.
*/
Calibration test_cal(void) {
    Exterior correct_ext = {
        105.2632, 102.7458, 403.8822,
        -0.2383291, 0.2442810, 0.0552577, 
        {{0.9688305, -0.0535899, 0.2418587}, 
        {-0.0033422, 0.9734041, 0.2290704},
        {-0.2477021, -0.2227387, 0.9428845}}};
    Interior correct_int = {-2.4742, 3.2567, 100.0000};
    Glass correct_glass = {0.0001, 0.00001, 150.0};
    ap_52 correct_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration correct_cal = {correct_ext, correct_int, correct_glass, 
        correct_addp};
    rotation_matrix(&correct_cal.ext_par);
    
    return correct_cal;
}

/* Fill in the fields of an Exterior struct to be all zeros. */
void zero_exterior(Exterior *ex) {
    int i, j;
    ex->x0 = ex->y0 = ex->z0 = ex->omega = ex->phi = ex->kappa = 0;
    
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            ex->dm[i][j] = 0;
}

/* Compare two 3x3 matrices. Return true if all elements are equal, 
   false otherwise
 */
int compare_matrix(Dmatrix m1, Dmatrix m2) {
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            if (fabs(m1[i][j] - m2[i][j]) > 1e-6) return 0;
    return 1;
}

/* Regression test for reading orientation files. Just reads a sample file and
   makes sure that nothing crashes and the orientation structures are filled
   out correctly.
*/
START_TEST(test_read_ori)
{
    Calibration correct_cal, *cal;
    correct_cal = test_cal();
    
    char ori_file[] = "testing_fodder/cal/cam1.tif.ori";
    char add_file[] = "testing_fodder/cal/cam1.tif.addpar";
    
    fail_if((cal = read_calibration(ori_file, add_file, NULL)) == NULL);
    fail_unless(compare_calib(cal, &correct_cal));
}
END_TEST

/* Unit test for writing orientation files. Writes a sample calibration,
   reads it back and compares.
*/
START_TEST(test_write_ori)
{
    Calibration correct_cal, *cal;
    correct_cal = test_cal();
    char ori_file[] = "testing_fodder/test.ori";
    char add_file[] = "testing_fodder/test.addpar";
    
    write_ori(correct_cal.ext_par, correct_cal.int_par,
        correct_cal.glass_par, correct_cal.added_par, ori_file, add_file);
    fail_if((cal = read_calibration(ori_file, add_file, NULL)) == NULL);
    fail_unless(compare_calib(cal, &correct_cal));
    
    remove(ori_file);
    remove(add_file);
}
END_TEST

START_TEST(test_rotation_angles)
{
    Exterior ex;
    /* Correct results may be verified using [1] */
    Dmatrix rotx = {{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}};
    Dmatrix roty = {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}};
    Dmatrix rotz = {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
    
    /* omega */
    zero_exterior(&ex);
    ex.omega = M_PI/2.;
    rotation_matrix(&ex);
    fail_unless(compare_matrix(ex.dm, rotx));
    
    /* phi */
    zero_exterior(&ex);
    ex.phi = M_PI/2.;
    rotation_matrix(&ex);
    fail_unless(compare_matrix(ex.dm, roty));
    
    /* kappa */
    zero_exterior(&ex);
    ex.kappa = M_PI/2.;
    rotation_matrix(&ex);
    fail_unless(compare_matrix(ex.dm, rotz));
}
END_TEST

Suite* ptv_suite(void) {
    Suite *s = suite_create ("PTV");
    TCase *tc;

    tc = tcase_create ("Read orientation file");
    tcase_add_test(tc, test_read_ori);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Write orientation file");
    tcase_add_test(tc, test_write_ori);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Basic rotations");
    tcase_add_test(tc, test_rotation_angles);
    suite_add_tcase (s, tc);
    
    return s;
}

int main(void) {
    int number_failed;
    Suite *s = ptv_suite ();
    SRunner *sr = srunner_create (s);
    srunner_run_all (sr, CK_ENV);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

