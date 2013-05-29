
/*  Unit-test suit for the C core of PyPTV. Uses the Check framework:
    http://check.sourceforge.net/
    
    To run it, type "make check" when in the top C directory, src_c/
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>

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
    rotation_matrix(correct_cal.ext_par, correct_cal.ext_par.dm);
    
    return correct_cal;
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

Suite* ptv_suite(void) {
    Suite *s = suite_create ("PTV");

    TCase *tc_rori = tcase_create ("Read orientation file");
    tcase_add_test(tc_rori, test_read_ori);
    suite_add_tcase (s, tc_rori);

    TCase *tc_wori = tcase_create ("Write orientation file");
    tcase_add_test(tc_wori, test_write_ori);
    suite_add_tcase (s, tc_wori);

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

