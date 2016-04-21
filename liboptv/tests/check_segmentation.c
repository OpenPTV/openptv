/* Unit tests for reading and writing parameter files. */

#include <check.h>
#include "segmentation.h"

START_TEST(test_peak_fit_new)
{
    char 
    filename_read[]  = "testing_fodder/parameters/targ_rec_all_different_fields.par",
    filename_write[] = "testing_fodder/parameters/targ_out_read.par";

    target_par targ_correct= { 
        .gvthres = {1, 2, 3, 4}, 
        .discont = 5,
        .nnmin = 6, .nnmax = 7,
        .nxmin = 8, .nxmax = 9,
        .nymin = 10, .nymax = 11, 
        .sumg_min = 12, 
        .cr_sz = 13 };
        
    int threshhold; 
    
    threshold = targ_correct.gvthres[0];
    
    target_par *targ_read = read_target_par(filename_read);
    fail_unless(compare_target_par(&targ_correct, targ_read));

    write_target_par(targ_read, filename_write);
    fail_unless(compare_target_par(&targ_correct, read_target_par(filename_write)));
    
    /* call peak_fit_new */
    peak_fit_new (unsigned char *img, targ_correct.gvthres[0], int xmin, int xmax, int ymin, 
int ymax, target pix[], control_par *cpar)

    remove(filename_write);
}
END_TEST

START_TEST(test_targ_rec)
{
    char
    filename_read[]  = "testing_fodder/parameters/targ_rec_all_different_fields.par",
    filename_write[] = "testing_fodder/parameters/targ_out_read.par";

    target_par targ_correct= { 
        .gvthres = {1, 2, 3, 4}, 
        .discont = 5,
        .nnmin = 6, .nnmax = 7,
        .nxmin = 8, .nxmax = 9,
        .nymin = 10, .nymax = 11, 
        .sumg_min = 12, 
        .cr_sz = 13 };
    
    target_par *targ_read = read_target_par(filename_read);
    fail_unless(compare_target_par(&targ_correct, targ_read));

    write_target_par(targ_read, filename_write);
    fail_unless(compare_target_par(&targ_correct, read_target_par(filename_write)));

    remove(filename_write);
}
END_TEST



Suite* fb_suite(void) {
    Suite *s = suite_create ("Peak fitting");

    TCase *tc = tcase_create ("check peak_fit_new");
    tcase_add_test(tc, test_peak_fit_new);
    suite_add_tcase (s, tc);
    
    
    tc = tcase_create ("Target recording");
    tcase_add_test(tc, test_targ_rec);
    suite_add_tcase (s, tc);


    return s;
}

int main(void) {
    int number_failed;
    Suite *s = fb_suite ();
    SRunner *sr = srunner_create (s);
    srunner_run_all (sr, CK_ENV);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

