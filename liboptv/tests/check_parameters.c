/* Unit tests for reading and writing parameter files. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include "parameters.h"

START_TEST(test_read_shaking_par)
{
	shaking_par * sp_test;
	shaking_par sp_correct={410000, 411055, 100, 3};	//according to current contents in shaking.par text file
	sp_test = read_shaking_par("testing_fodder/parameters/shaking.par");

	fail_unless(compare_shaking_par(sp_test, &sp_correct));

}
END_TEST

START_TEST(test_read_sequence_par)
{
    int cam;
    char fname[SEQ_FNAME_MAX_LEN];
    sequence_par *seqp;

    seqp = read_sequence_par("testing_fodder/parameters/sequence.par");
    
    for (cam = 0; cam < 4; cam++) {
        printf("%s", seqp->img_base_name[cam]);
        sprintf(fname, "dumbbell/cam%d_Scene77_", cam + 1);
        fail_unless(strncmp(fname, seqp->img_base_name[cam],
            SEQ_FNAME_MAX_LEN - 1) == 0);
    }
    fail_unless(seqp->first == 497);
    fail_unless(seqp->last == 597);
    
    
    
}
END_TEST

START_TEST(test_read_track_par)
{
    track_par tpar_correct = {
        0.4, 120, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 0., 0., 0., 0., 1. 
    };
    
    track_par *tpar;
    tpar = read_track_par("testing_fodder/parameters/track.par");
    
    fail_unless(compare_track_par(tpar, &tpar_correct));
}
END_TEST

START_TEST(test_read_volume_par)
{
    volume_par vpar_correct = {
        {-250., 250.}, {-100., -100.}, {100., 100.}, 0.01, 0.3, 0.3, 0.01, 1, 33
    };
    
    volume_par *vpar;
    vpar = read_volume_par("testing_fodder/parameters/criteria.par");
    
    fail_unless(compare_volume_par(vpar, &vpar_correct));
}
END_TEST

START_TEST(test_read_control_par)
{
    int cam;
    char img_format[] = "dumbbell/cam%d_Scene77_4085";
    char cal_format[] = "cal/cam%d.tif";
    control_par cpar_correct, *cpar;
    
    cpar_correct.num_cams = 4;
    cpar_correct.img_base_name = (char **) malloc(4*sizeof(char *));
    cpar_correct.cal_img_base_name = (char **) malloc(4*sizeof(char *));
    cpar_correct.mm = (mm_np *) malloc(sizeof(mm_np));
    
    
    for (cam = 0; cam < 4; cam++) {
        cpar_correct.img_base_name[cam] = 
            (char *) malloc((strlen(img_format) + 1) * sizeof(char));
        sprintf(cpar_correct.img_base_name[cam], img_format, cam + 1);
        
        cpar_correct.cal_img_base_name[cam] = 
            (char *) malloc((strlen(cal_format) + 1) * sizeof(char));
        sprintf(cpar_correct.cal_img_base_name[cam], cal_format, cam + 1);
    }
    
    cpar_correct.hp_flag = 1;
    cpar_correct.allCam_flag = 0;
    cpar_correct.tiff_flag = 1;
    cpar_correct.imx = 1280;
    cpar_correct.imy = 1024;
    cpar_correct.pix_x  = 0.017;
    cpar_correct.pix_y = 0.017;
    cpar_correct.chfield = 0;
    cpar_correct.mm->n1 = 1;
    cpar_correct.mm->n2[0] = 1.49;
    cpar_correct.mm->n3 = 1.33;
    cpar_correct.mm->d[0] = 5;
    
  
    cpar = read_control_par("testing_fodder/parameters/ptv.par");
    fail_unless(compare_control_par(cpar, &cpar_correct));
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("Parameters handling");
    TCase *tc;

    tc=tcase_create ("Read shaking parameters");
    tcase_add_test(tc, test_read_shaking_par);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Read sequence parameters");
    tcase_add_test(tc, test_read_sequence_par);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Read tracking parameters");
    tcase_add_test(tc, test_read_track_par);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Read illuminated volume parameters");
    tcase_add_test(tc, test_read_volume_par);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Read control parameters");
    tcase_add_test(tc, test_read_control_par);
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

