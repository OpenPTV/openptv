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
#include "trafo.h"

START_TEST(test_ray_distance_midpoint)
{
    /* Generate simply-oriented skew rays with a known distance and get that
       distance from the distance finding routine. */
    
    vec3d pos1 = {0., 0., 0.}, dir1 = {1., 0., 0.};
    vec3d pos2 = {0., 0., 1.}, dir2 = {0., 1., 0.};
    vec3d midpoint, skew_midp = {0., 0., 0.5};
    
    /* Skew rays case: */
    fail_unless(skew_midpoint(pos1, dir1, pos2, dir2, midpoint) == 1.);
    fail_unless(vec_cmp(midpoint, skew_midp));
    
    /* Intersecting rays case: */
    fail_unless(skew_midpoint(pos1, dir1, dir1, dir2, midpoint) == 0.);
    fail_unless(vec_cmp(midpoint, dir1));
}
END_TEST

START_TEST(test_point_position)
{
    /* Generate target information based on a known point. In the simple case,
       calculated point must equal known point and average ray distance == 0.
       Then jigg the cameras symmetrically to get the same points again but 
       with an analytically derivable ray distance. */
    
    int cam, num_cams = 4;
    vec2d targs_plain[4], targs_jigged[4];
    
    Calibration *calib[4];
    char ori_tmpl[] = "cal/sym_cam%d.tif.ori";
    char ori_name[25];
    mm_np media_par = {1, 1., {1., 0., 0.}, {1., 0., 0.}, 1., 1.};
    
    vec3d point = {17, 42, 0}; /* Something in the FOV, non-symmetric. */
    vec3d res, jigged;
    double jigg_amp = 0.5, skew_dist, jigged_correct;
    
    /* Target generation requires an existing calibration. */
    chdir("testing_fodder/");
    
    /* Four cameras on 4 quadrants. */
    for (cam = 0; cam < num_cams; cam++) {
        sprintf(ori_name, ori_tmpl, cam + 1);
        calib[cam] = read_calibration(ori_name, "cal/cam1.tif.addpar", NULL);
        
        img_coord(point, calib[cam], &media_par, 
            &(targs_plain[cam][0]), &(targs_plain[cam][1]));
        
        vec_copy(jigged, point);
        jigged[1] += ((cam % 2) ? jigg_amp : -jigg_amp);
        img_coord(jigged, calib[cam], &media_par, 
            &(targs_jigged[cam][0]), &(targs_jigged[cam][1]));
    }
    
    skew_dist = point_position(targs_plain, num_cams, &media_par, calib, res);
    fail_unless(skew_dist < 1e-10);
    vec_subt(point, res, res);
    fail_unless(vec_norm(res) < 1e-10);
    
    skew_dist = point_position(targs_jigged, num_cams, &media_par, calib, res);
    jigged_correct = 4*(2*jigg_amp)/6;
    /* two thirds, but because of details: each disagreeing pair (4 out of 6) 
       has a 2*jigg_amp error because the cameras are moved in opposite
       directions. */
    fail_unless(fabs(skew_dist - jigged_correct) < 0.05);
    vec_subt(point, res, res);
    fail_unless(vec_norm(res) < 0.01);
}
END_TEST

START_TEST(test_convergence_measure)
{
    /* Generate target information based on known points. In the simple case,
       convergence will be spot-on. Then generate other targets based on a 
       per-camera jigging of 3D points and get convergence measure compatible 
       with the jig amplitude. */
    
    vec3d known[16], jigged;
    vec2d* targets[16];
    Calibration *calib[4];
    
    int num_cams = 4, num_pts = 16;
    int cam, cpt_vert, cpt_horz, cpt_ix;
    double jigg_amp = 0.5, jigged_skew_dist, jigged_correct;
    
    char ori_tmpl[] = "cal/sym_cam%d.tif.ori";
    char ori_name[25];
    
    /* Using a neutral medium, this isn't what's tested. */
    mm_np media_par = {1, 1., {1., 0., 0.}, {1., 0., 0.}, 1., 1.};
    control_par *cpar;
    
    /* Target generation requires an existing calibration. */
    chdir("testing_fodder/");
    cpar = read_control_par("parameters/ptv.par");
    
    /* Four cameras on 4 quadrants looking down into a calibration target.
       Calibration taken from an actual experimental setup */
    for (cam = 0; cam < num_cams; cam++) {
        sprintf(ori_name, ori_tmpl, cam + 1);
        calib[cam] = read_calibration(ori_name, "cal/cam1.tif.addpar", NULL);
    }
    
    /* Reference points before jigging: */
    for (cpt_horz = 0; cpt_horz < 4; cpt_horz++) {
        for (cpt_vert = 0; cpt_vert < 4; cpt_vert++) {
            cpt_ix = cpt_horz*4 + cpt_vert;
            vec_set(known[cpt_ix], cpt_vert * 10, cpt_horz * 10, 0);
        }
    }
    
    /* Plain case: */
    for (cpt_ix = 0; cpt_ix < num_pts; cpt_ix++) {
        targets[cpt_ix] = (vec2d *) calloc(num_cams, sizeof(vec2d));
        
        for (cam = 0; cam < num_cams; cam++) {
            img_coord(known[cpt_ix], calib[cam], &media_par, 
                &(targets[cpt_ix][cam][0]), 
                &(targets[cpt_ix][cam][1]));
        }
    }
    fail_unless(fabs(weighted_dumbbell_precision(
        targets, 16, num_cams, &media_par, calib, 1, 0)) < 1e-10);
    /* With dumbbell length: */
    fail_unless(fabs(weighted_dumbbell_precision(
        targets, 16, num_cams, &media_par, calib, 10, 10)) < 1e-10);
    
    /* Jigged case (reusing target memory), moving the points and cameras 
       in parallel to create a known shift between parallel rays. */
    for (cam = 0; cam < num_cams; cam++) {
        calib[cam]->ext_par.y0 += ((cam % 2) ? jigg_amp : -jigg_amp);
        
        for (cpt_ix = 0; cpt_ix < num_pts; cpt_ix++) {
            vec_copy(jigged, known[cpt_ix]);
            jigged[1] += ((cam % 2) ? jigg_amp : -jigg_amp);
            
            img_coord(jigged, calib[cam], &media_par, 
                &(targets[cpt_ix][cam][0]), 
                &(targets[cpt_ix][cam][1]));
        }
    }
    jigged_skew_dist = weighted_dumbbell_precision(
        targets, 16, num_cams, &media_par, calib, 1, 0);
    jigged_correct = 16*4*(2*jigg_amp)/(16*6); 
    /* two thirds, but because of details: each disagreeing pair (4 out of 6) 
       has a 2*jigg_amp error because the cameras are moved in opposite
       directions. */
    fail_unless(fabs(jigged_skew_dist - jigged_correct) < 0.05 );
}
END_TEST


Suite* orient_suite(void) {
    Suite *s = suite_create ("Finding calibration parameters");

    TCase *tc = tcase_create ("Skew rays");
    tcase_add_test(tc, test_ray_distance_midpoint);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Point position");
    tcase_add_test(tc, test_point_position);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Convergence measures");
    tcase_add_test(tc, test_convergence_measure);
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

