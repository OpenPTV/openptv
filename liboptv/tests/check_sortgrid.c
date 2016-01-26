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
    int num_points, correct_num_points = 6;
    vec3d fix[100];
    char calblock_file[] = "testing_fodder/cal/calibration_block.txt";
    

    ck_assert_msg (file_exists(calblock_file) == 1, 
        "\n File %s does not exist\n", calblock_file);
    num_points = read_calblock(fix, calblock_file);   
    fail_if (num_points == 0, "\n calblock file reading failed \n");

    fail_unless(num_points == correct_num_points);
    

}
END_TEST

START_TEST(test_sortgrid)
{
    Calibration *cal;
    control_par *cpar;
    vec3d fix[100];
    target pix[20], short_pix[2];
    target *sorted_pix, *short_sorted_pix;
    int nfix, i;
    int eps, correct_eps = 25;

    eps = read_sortgrid_par("testing_fodder/parameters/sortgrid.par");
    fail_unless(eps == correct_eps);


    char *file_base = "testing_fodder/sample_";
    int targets_read = 0;

    /* sample_0044_targets is for the case nfix < num */
    targets_read = read_targets(pix, file_base, 44);
    fail_unless(targets_read == 8);


    char ori_file[] = "testing_fodder/cal/cam1.tif.ori";
    char add_file[] = "testing_fodder/cal/cam1.tif.addpar";

    fail_if((cal = read_calibration(ori_file, add_file, NULL)) == 0);

    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par")) == 0);

    fail_if((nfix = read_calblock(fix,"testing_fodder/cal/calibration_block.txt")) != 6);   

    /* nothing should be assigned, the search region is 0.0 */
    sortgrid (cal, cpar, nfix, fix, targets_read, 0.0, pix);
    fail_unless(pix[0].pnr == -999);
    fail_unless(pix[targets_read-1].pnr == -999);
     
    sortgrid (cal, cpar, nfix, fix, targets_read, eps, pix);
    fail_unless(pix[2].pnr == 5);
    fail_unless(pix[1].pnr == 3);

    /* using very large region of search ensures some matches */
    sortgrid (cal, cpar, nfix, fix, targets_read, 120, pix);
    fail_unless(pix[2].pnr == 5);
    fail_unless(pix[1].pnr == 3);
    fail_unless(pix[4].pnr == 1);
    
    /* sample_0043_targets is for the case nfix > num */ 
    targets_read = read_targets(short_pix, file_base, 43);
    fail_unless(targets_read == 2); 

    sortgrid (cal, cpar, nfix, fix, targets_read, 120, short_pix);
    fail_unless(short_pix[1].pnr == 5);


    
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("Sortgrid");

    
    TCase *tc = tcase_create ("Read sortgrid.par");
    tcase_add_test(tc, test_read_sortgrid_par);     
    suite_add_tcase (s, tc); 
    
    tc = tcase_create ("Read calblock");
    tcase_add_test(tc, test_read_calblock);     
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


int file_exists(char *filename){
    if( access(filename, F_OK ) != -1 ) {
        return 1;
    } else {
        printf("File %s does not exist\n",filename);
        return 0;
    }
}

