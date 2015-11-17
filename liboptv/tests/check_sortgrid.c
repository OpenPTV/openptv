/* Unit tests for finding image coordinates of 3D position. */

#include <check.h>
#include <stdlib.h>
#include <math.h>

#include "sortgrid.h"
#include "parameters.h"
#include "tracking_frame_buf.h"

#define EPS 1E-6

START_TEST(test_nearest_neighbour_pix)
{
    target t1 = {0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1};
    int pnr = -999;
    
    /* test for zero distance */
    pnr = nearest_neighbour_pix (&t1, 1, 1128.0, 795.0, 0.0);
    fail_unless(pnr == -999);
    
    /* test for negative epsilon */
    pnr = nearest_neighbour_pix (&t1, 1, 1128.0, 795.0, -1.0);
    fail_unless(pnr == -999);
        
    /* test for negative pixel values */
    pnr = nearest_neighbour_pix (&t1, 1, -1127.0, -796.0, 1E3);
    fail_unless(pnr == -999);
    
    /* test for the correct use */
    pnr = nearest_neighbour_pix (&t1, 1, 1127.0, 796.0, 1E-5);
    fail_unless(pnr == 0);
    
}
END_TEST

START_TEST(test_read_sortgrid_par)
{
    fail_unless (NULL);
}
END_TEST

START_TEST(test_sortgrid)
{
    fail_unless (NULL);
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("Sortgrid");
 
    TCase *tc = tcase_create ("Nearest neighbour search");
    tcase_add_test(tc, test_nearest_neighbour_pix);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Read sortgrid.par");
    tcase_add_test(tc, test_read_sortgrid_par);     
    suite_add_tcase (s, tc); 
      
    tc = tcase_create ("Sortgrid");
    tcase_add_test(tc, test_sortgrid);     
    suite_add_tcase (s, tc); 
//     
//     tc = tcase_create ("Distorted image coordinates");
//     tcase_add_test(tc, test_distorted_centered_cam);
//     suite_add_tcase (s, tc);
//     
//     tc = tcase_create ("Shifted sensor not ignored");
//     tcase_add_test(tc, test_shifted_sensor);
//     suite_add_tcase (s, tc);
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

