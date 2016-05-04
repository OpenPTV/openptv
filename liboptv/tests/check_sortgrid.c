/* Unit tests for finding image coordinates of 3D position. */

#include <check.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>


#include "sortgrid.h"
#include "parameters.h"
#include "tracking_frame_buf.h"

#define EPS 1E-6

int file_exists();

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
    int eps, correct_eps = 25;

    eps = read_sortgrid_par("testing_fodder/parameters/sortgrid.par");
    fail_unless(eps == correct_eps);
    
    eps = read_sortgrid_par("testing_fodder/parameters/sortgrid_corrupted.par");
    fail_unless(eps == 0);

}
END_TEST

START_TEST(test_read_calblock)
{
    int num_points, correct_num_points = 5;
    vec3d *fix;
    char calblock_file[] = "testing_fodder/cal/calblock.txt";
    
    ck_assert_msg (file_exists(calblock_file) == 1, 
        "\n File %s does not exist\n", calblock_file);
    
    fix = read_calblock(&num_points, calblock_file);   
    
    fail_if (num_points == 0, "\n calblock file reading failed \n");
    fail_unless(num_points == correct_num_points);
}
END_TEST

START_TEST(test_sortgrid)
{
    Calibration *cal;
    control_par *cpar;
    vec3d *fix;
    target pix[2];
    target *sorted_pix;
    int nfix, i;
    int eps, correct_eps = 25;

    eps = read_sortgrid_par("testing_fodder/parameters/sortgrid.par");
    fail_unless(eps == correct_eps);


    char *file_base = "testing_fodder/sample_";
    int frame_num = 42;
    int targets_read = 0;

    targets_read = read_targets(pix, file_base, frame_num);
    fail_unless(targets_read == 2);


    char ori_file[] = "testing_fodder/cal/cam1.tif.ori";
    char add_file[] = "testing_fodder/cal/cam1.tif.addpar";

    fail_if((cal = read_calibration(ori_file, add_file, NULL)) == 0);
    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par")) == 0);
    fail_if((fix = read_calblock(&nfix,"testing_fodder/cal/calblock.txt")) == NULL);
    fail_unless(nfix == 5);

    sorted_pix = sortgrid (cal, cpar, nfix, fix, targets_read, eps, pix);
    fail_unless(sorted_pix[0].pnr == -999);
    fail_unless(sorted_pix[1].pnr == -999);

    sorted_pix = sortgrid (cal, cpar, nfix, fix, targets_read, 120, pix);
    fail_unless(sorted_pix[1].pnr == 1);
    fail_unless(sorted_pix[1].x == 796);
    
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
    
    tc = tcase_create ("Read calblock");
    tcase_add_test(tc, test_read_calblock);     
    suite_add_tcase (s, tc);
      
    tc = tcase_create ("Sortgrid");
    tcase_add_test(tc, test_sortgrid);     
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


int file_exists(char *filename){
    if( access(filename, F_OK ) != -1 ) {
        return 1;
    } else {
        printf("File %s does not exist\n",filename);
        return 0;
    }
}

