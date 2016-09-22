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

#include "calibration.h"
#include "vec_utils.h"
#include "parameters.h"
#include "imgcoord.h"
#include "trafo.h"
#include "correspondences.h"

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

START_TEST(test_quicksort_target_y)
{
    int num; 
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
    num = sizeof(test_pix)/sizeof(test_pix[0]);
    quicksort_target_y (test_pix, num);

    /* first point should be -1.1 and the last 0.8 */
    fail_unless (fabs(test_pix[0].y + 1.1) < 1E-6);
    fail_unless (fabs(test_pix[1].y + 0.2) < 1E-6);
    fail_unless (fabs(test_pix[num-1].y - 0.8) < 1E-6);
    

    
}
END_TEST

START_TEST(test_quicksort_coord2d_x)
{
    			   
    int num = 7; /* length of the test_crd */

    /* coord_2d is int pnr, double x,y */
    coord_2d test_crd[] = {
        {0, 0.0, 0.0},
        {6, 0.1, 0.1}, /* best candidate, right on the diagonal */
        {3, 0.2, -0.8},
        {4, -0.4, -1.1},
        {1, 0.7, -0.1},
        {7, 1.2, 0.3},
        {5, 10.4, 0.1}
    };

    quicksort_coord2d_x(test_crd, num);
   
    /* first point should be -0.4 and the last 10.4 */
    fail_unless (fabs(test_crd[0].x + 0.4) < 1E-6);
    fail_unless (fabs(test_crd[1].x - 0.0) < 1E-6);
    fail_unless (fabs(test_crd[num-1].x - 10.4) < 1E-6);	
}
END_TEST

START_TEST(test_quicksort_con)
{
    			   
    int i, num = 3; /* length of the test_crd */
    
    n_tupel test_con[] = {
        {{0, 1, 2, 3}, 0.1},
        {{0, 1, 2, 3}, 0.2},
        {{0, 1, 2, 3}, 0.15}
    };

    quicksort_con(test_con, num);

   
    /* first point should be -0.4 and the last 10.4 */
    fail_unless (fabs(test_con[0].corr - 0.2) < 1E-6);
    fail_unless (fabs(test_con[2].corr - 0.1) < 1E-6);
   
	
}
END_TEST

/* Tests of correspondence components and full process using dummy data */

void read_all_calibration(Calibration *calib[4], control_par *cpar) {
    char ori_tmpl[] = "testing_fodder/cal/sym_cam%d.tif.ori";
    char added_name[] = "testing_fodder/cal/cam1.tif.addpar";
    char ori_name[40];
    int cam;
    
    for (cam = 0; cam < cpar->num_cams; cam++) {
        sprintf(ori_name, ori_tmpl, cam + 1);
        calib[cam] = read_calibration(ori_name, added_name, NULL);
    }
}

/* generate_test_set() generates data for targets on 4 cameras. The targets
 * Are organized on a 4x4 grid, 10mm apart.
 */
frame *generate_test_set(Calibration *calib[4], control_par *cpar, 
    volume_par *vpar)
{
    int cam, cpt_horz, cpt_vert, cpt_ix;
    target *targ;
    vec3d tmp;
    frame *frm = (frame *) malloc(sizeof(frame));
    
    /* Four cameras on 4 quadrants looking down into a calibration target.
       Calibration taken from an actual experimental setup */
    frame_init(frm, cpar->num_cams, 16);
    
    for (cam = 0; cam < cpar->num_cams; cam++) {
        frm->num_targets[cam] = 16;
        
        /* Construct a scene representing a calibration target, generate
           tergets for it, then use them to reconstruct correspondences. */
        for (cpt_horz = 0; cpt_horz < 4; cpt_horz++) {
            for (cpt_vert = 0; cpt_vert < 4; cpt_vert++) {
                cpt_ix = cpt_horz*4 + cpt_vert;
                if (cam % 2) cpt_ix = 15 - cpt_ix; /* Avoid symmetric case */
                
                targ = &(frm->targets[cam][cpt_ix]);
                targ->pnr = cpt_ix;
                
                vec_set(tmp, cpt_vert * 10, cpt_horz * 10, 0);
                img_coord(tmp, calib[cam], cpar->mm, &(targ->x), &(targ->y));
                metric_to_pixel(&(targ->x), &(targ->y), targ->x, targ->y, cpar);
                
                /* These values work in check_epi, so used here too */
                targ->n = 25;
                targ->nx = targ->ny = 5;
                targ->sumg = 10;
            }
        }
    }
    return frm;
}

/*  correct_frame() performs the transition from pixel to metric to flat 
    coordinates and x-sorting as required by the correspondence code.
    
    Arguments:
    frame *frm - target information for all cameras.
    control_par *cpar - parameters of image size, pixel size etc.
    tol - tolerance parameter for iterative flattening phase, see 
        trafo.h:correct_brown_affine_exact().
*/
coord_2d **correct_frame(frame *frm, Calibration *calib[], control_par *cpar, 
    double tol) 
{
    coord_2d **corrected;
    int cam, part;
    
    corrected = (coord_2d **) malloc(cpar->num_cams * sizeof(coord_2d *));
    for (cam = 0; cam < cpar->num_cams; cam++) {
        corrected[cam] = (coord_2d *) malloc(
            frm->num_targets[cam] * sizeof(coord_2d));
        if (corrected[cam] == NULL) {
            /* roll back allocations and fail */
            for (cam -= 1; cam >= 0; cam--) free(corrected[cam]);
            free(corrected);
            return NULL;
        }
        
        for (part = 0; part < frm->num_targets[cam]; part++) {
            pixel_to_metric(&corrected[cam][part].x, &corrected[cam][part].y,
                          frm->targets[cam][part].x, frm->targets[cam][part].y,
                          cpar);
            
            dist_to_flat(corrected[cam][part].x, corrected[cam][part].y,
                calib[cam], &corrected[cam][part].x, &corrected[cam][part].y,
                tol);
        
            corrected[cam][part].pnr = frm->targets[cam][part].pnr;
        }
        
        /* This is expected by find_candidate() */
        quicksort_coord2d_x(corrected[cam], frm->num_targets[cam]);
    }
    return corrected;
}

START_TEST(test_pairwise_matching)
{
    frame *frm;
    Calibration *calib[4];
    volume_par *vpar;
    control_par *cpar;
    correspond *list[4][4];
    coord_2d **corrected;
    int cam, subcam, part, cand, correct_pnr;
    
    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par"))== 0);
    fail_if((vpar = read_volume_par("testing_fodder/parameters/criteria.par"))==0);
    
    /* Cameras are at so high angles that opposing cameras don't see each other
       in the normal air-glass-water setting. */
    cpar->mm->n2[0] = 1.0001;
    cpar->mm->n3 = 1.0001;
    
    read_all_calibration(calib, cpar);
    frm = generate_test_set(calib, cpar, vpar);
    
    corrected = correct_frame(frm, calib, cpar, 0.0001);
    safely_allocate_adjacency_lists(list, cpar->num_cams, frm->num_targets);
    
    match_pairs(list, corrected, frm, vpar, cpar, calib);
    
    /* Well, I guess we should at least check that each target has the 
       real matches as candidates, as a sample check. */
    for (cam = 0; cam < cpar->num_cams - 1; cam++) {
        for (subcam = cam + 1; subcam < cpar->num_cams; subcam++) {
            for (part = 0; part < frm->num_targets[cam]; part++) {
                
                /* Complications here:
                   1. target numbering scheme alternates.
                   2. Candidte 'pnr' is an index into the x-sorted array, not
                      the original pnr.
                */
                if ((subcam - cam) % 2 == 0) {
                    correct_pnr = corrected[cam][list[cam][subcam][part].p1].pnr;
                } else {
                    correct_pnr = 15 - corrected[cam][list[cam][subcam][part].p1].pnr;
                }
                
                for (cand = 0; cand < MAXCAND; cand++) {
                    if (corrected[subcam][list[cam][subcam][part].p2[cand]].pnr == correct_pnr) break;
                }
                fail_if(cand == MAXCAND);
            }
        }
    }
    
    /* Clean up */
    for (cam = 0; cam < cpar->num_cams; cam++) 
        free(corrected[cam]);
    free(corrected);
    deallocate_adjacency_lists(list, cpar->num_cams);
}
END_TEST

START_TEST(test_four_camera_matching)
{
    frame *frm;
    Calibration *calib[4];
    volume_par *vpar;
    control_par *cpar;
    correspond *list[4][4];
    coord_2d **corrected;
    n_tupel *con;
    int cam, matched;
    
    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par"))== 0);
    fail_if((vpar = read_volume_par("testing_fodder/parameters/criteria.par"))==0);
    
    /* Cameras are at so high angles that opposing cameras don't see each other
       in the normal air-glass-water setting. */
    cpar->mm->n2[0] = 1.0001;
    cpar->mm->n3 = 1.0001;
    
    read_all_calibration(calib, cpar);
    frm = generate_test_set(calib, cpar, vpar);

    corrected = correct_frame(frm, calib, cpar, 0.0001);
    safely_allocate_adjacency_lists(list, cpar->num_cams, frm->num_targets);
    match_pairs(list, corrected, frm, vpar, cpar, calib);
    
    con = (n_tupel *) malloc(16*sizeof(n_tupel));
    matched = four_camera_matching(list, 16, 1., con, 16);

    fail_unless(matched == 16);
    
    /* Clean up */
    for (cam = 0; cam < cpar->num_cams; cam++) 
        free(corrected[cam]);
    deallocate_adjacency_lists(list, cpar->num_cams);
    free(con);
}
END_TEST

START_TEST(test_three_camera_matching)
{
    /* the overall setup is the same as the 4-camera test, with the following
       changes: targets are darkenned in one camera to get 3*16 triplets;
       one target is marked as used to reduce it to 15 trips.*/
    frame *frm;
    target *targ;
    
    Calibration *calib[4];
    volume_par *vpar;
    control_par *cpar;
    
    correspond *list[4][4];
    coord_2d **corrected;
    n_tupel *con;
    int **tusage;
    
    int cam, matched, part;
    
    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par"))== 0);
    fail_if((vpar = read_volume_par("testing_fodder/parameters/criteria.par"))==0);
    
    cpar->mm->n2[0] = 1.0001;
    cpar->mm->n3 = 1.0001;
    
    read_all_calibration(calib, cpar);
    frm = generate_test_set(calib, cpar, vpar);
    
    /* Darken cam 2 below acceptance.*/
    for (part = 0; part < frm->num_targets[1]; part++) {
        targ = &(frm->targets[1][part]);
        targ->n = 0;
        targ->nx = targ->ny = 0;
        targ->sumg = 0;
    }

    corrected = correct_frame(frm, calib, cpar, 0.0001);
    safely_allocate_adjacency_lists(list, cpar->num_cams, frm->num_targets);
    match_pairs(list, corrected, frm, vpar, cpar, calib);
    
    con = (n_tupel *) malloc(4*16*sizeof(n_tupel));
    tusage = safely_allocate_target_usage_marks(cpar->num_cams);
    
    /* high accept corr bcz of closeness to epipolar lines. */
    matched = three_camera_matching(list, 4, frm->num_targets, 100000., con, 
        4*16, tusage);
    
    fail_unless(matched == 16);
    
    /* Clean up */
    for (cam = 0; cam < cpar->num_cams; cam++) 
        free(corrected[cam]);
    deallocate_adjacency_lists(list, cpar->num_cams);
    deallocate_target_usage_marks(tusage, cpar->num_cams);
    free(con);
    free(vpar);
    free_control_par(cpar);
}
END_TEST

START_TEST(test_two_camera_matching)
{
    /* the overall setup is the same as the 4-camera test, with the following
       changes: targets are darkenned in two cameras to get 16 pairs. */
    frame *frm;
    target *targ;
    
    Calibration *calib[4];
    volume_par *vpar;
    control_par *cpar;
    
    correspond *list[4][4];
    coord_2d **corrected;
    n_tupel *con;
    int **tusage;
    
    int cam, matched, part;
    
    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par"))== 0);
    fail_if((vpar = read_volume_par("testing_fodder/parameters/criteria.par"))==0);
    
    cpar->mm->n2[0] = 1.0001;
    cpar->mm->n3 = 1.0001;
    vpar->Zmin_lay[0] = -1;
    vpar->Zmin_lay[1] = -1;
    vpar->Zmax_lay[0] = 1;
    vpar->Zmax_lay[1] = 1;
    
    read_all_calibration(calib, cpar);
    frm = generate_test_set(calib, cpar, vpar);
    
    cpar->num_cams = 2;
    corrected = correct_frame(frm, calib, cpar, 0.0001);
    safely_allocate_adjacency_lists(list, cpar->num_cams, frm->num_targets);
    match_pairs(list, corrected, frm, vpar, cpar, calib);
    
    con = (n_tupel *) malloc(4*16*sizeof(n_tupel));
    tusage = safely_allocate_target_usage_marks(cpar->num_cams);
    
    /* high accept corr bcz of closeness to epipolar lines. */
    matched = consistent_pair_matching(list, 2, frm->num_targets, 10000., con, 
        4*16, tusage);
    
    fail_unless(matched == 16);
    
    /* Clean up */
    for (cam = 0; cam < cpar->num_cams; cam++) 
        free(corrected[cam]);
    deallocate_adjacency_lists(list, cpar->num_cams);
    deallocate_target_usage_marks(tusage, cpar->num_cams);
    free(con);
    free(vpar);
    free_control_par(cpar);
}
END_TEST

START_TEST(test_correspondences)
{
    frame *frm;
    Calibration *calib[4];
    volume_par *vpar;
    control_par *cpar;
    coord_2d **corrected;
    n_tupel *con;
    int cam, match_counts[4];
    
    fail_if((cpar = read_control_par("testing_fodder/parameters/ptv.par"))== 0);
    fail_if((vpar = read_volume_par("testing_fodder/parameters/criteria.par"))==0);
    
    /* Cameras are at so high angles that opposing cameras don't see each other
       in the normal air-glass-water setting. */
    cpar->mm->n2[0] = 1.0001;
    cpar->mm->n3 = 1.0001;
        
    read_all_calibration(calib, cpar);
    frm = generate_test_set(calib, cpar, vpar);
    corrected = correct_frame(frm, calib, cpar, 0.0001);
    con = correspondences(frm, corrected, vpar, cpar, calib, match_counts);
    
    /* The example set is built to have all 16 quadruplets. */
    fail_unless(match_counts[0] == 16);
    fail_unless(match_counts[1] == 0);
    fail_unless(match_counts[2] == 0);
    fail_unless(match_counts[3] == 16); /* last elemnt is the sum of matches */
}
END_TEST


Suite* corresp_suite(void) {
    Suite *s = suite_create ("Testing correspondences");

    TCase *tc = tcase_create ("qs target y");
    tcase_add_test(tc, test_qs_target_y);
    suite_add_tcase (s, tc);

    tc = tcase_create ("quicksort_target_y");
    tcase_add_test(tc, test_quicksort_target_y);
    suite_add_tcase (s, tc);

    tc = tcase_create ("quicksort_coord2d_x");
    tcase_add_test(tc, test_quicksort_coord2d_x);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("quicksort_con");
    tcase_add_test(tc, test_quicksort_con);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Calibration target correspondences");
    tcase_add_test(tc, test_correspondences);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Pairwise matching");
    tcase_add_test(tc, test_pairwise_matching);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("four camera matching");
    tcase_add_test(tc, test_four_camera_matching);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Three camera matching");
    tcase_add_test(tc, test_three_camera_matching);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Two camera matching");
    tcase_add_test(tc, test_two_camera_matching);
    suite_add_tcase (s, tc);

    return s;
}

int main(void) {
    int number_failed;
    Suite *s = corresp_suite();
    SRunner *sr = srunner_create (s);
    srunner_run_all (sr, CK_ENV);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
