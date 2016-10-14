/*  Unit tests for the tracking. Uses the Check
    framework: http://check.sourceforge.net/

    To run it, type "make verify" when in the top C directory, src_c/
    If that doesn't run the tests, use the Check tutorial:
    http://check.sourceforge.net/doc/check_html/check_3.html
*/

/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "track.h"
#include "calibration.h"

#define EPS 1E-5

void read_all_calibration(Calibration *calib[3], int num_cams);
void create_tracking_test_case();

void create_tracking_test_case()
{
    Calibration *calib[3];
    int step, test_step = 10001;
    vec3d point = {0.0, 0.0, 0.0};
    vec2d v[3];
    control_par *cpar;
    
    
    chdir("testing_fodder/track");
    
    /* prepare the 3D point moving in the flow */
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    
    target t1 = {0, 1127.0000, 796.0000, 10, 2, 2, 100, 0};
    
    char target_tmpl[] = "img/cam%d.";
    char target_name[256];
    
    for(step = test_step-1; step < test_step + 4; step++){
        for (int j=0; j<cpar->num_cams;j++){
            point[0] += 0.1;
            point_to_pixel(v[j], point, calib[j], cpar);
            t1.x = v[j][0]; t1.y = v[j][1];
            sprintf(target_name, target_tmpl, j + 1);
            write_targets(&t1, 1, target_name, step);
        }
    }
}


/* Tests of correspondence components and full process using dummy data */

void read_all_calibration(Calibration *calib[3], int num_cams) {
    char ori_tmpl[] = "cal/cam%d.tif.ori";
    char added_tmpl[] = "cal/cam%d.tif.addpar";
    char ori_name[256],added_name[256];
    int cam;

    for (cam = 0; cam < num_cams; cam++) {
        sprintf(ori_name, ori_tmpl, cam + 1);
        sprintf(added_name, added_tmpl, cam + 1);
        calib[cam] = read_calibration(ori_name, added_name, NULL);
    }
}


START_TEST(test_predict)
{
    vec2d prev_pos = {1.1, 0.6};
    vec2d curr_pos = {2.0, -0.8};
    vec2d result = {2.9, -2.2};

    vec2d c;

    predict(prev_pos,curr_pos,c);


    ck_assert_msg( fabs(c[0] - result[0]) < EPS,
             "Was expecting 2.9 but found %f \n", fabs(c[0]));

    ck_assert_msg( fabs(c[1] - result[1]) < EPS,
             "Was expecting -2.2 but found %f \n", fabs(c[1]));

}
END_TEST


START_TEST(test_search_volume_center_moving)
{
    vec3d prev_pos = {1.1, 0.6, 0.1};
    vec3d curr_pos = {2.0, -0.8, 0.2};
    vec3d result = {2.9, -2.2, 0.3};

    vec3d c;

    search_volume_center_moving(prev_pos, curr_pos, c);


    ck_assert_msg( fabs(c[0] - result[0]) < EPS,
             "Was expecting 2.9 but found %f \n", c[0]);

    ck_assert_msg( fabs(c[1] - result[1]) < EPS,
             "Was expecting -2.2 but found %f \n", c[1]);

    ck_assert_msg( fabs(c[2] - result[2]) < EPS,
                      "Was expecting 0.3 but found %f \n", c[2]);

}
END_TEST

START_TEST(test_pos3d_in_bounds)
{
    vec3d inside = {1.0,-1.0,0.0};
    vec3d outside = {2.0, -0.8, 2.1};

    track_par bounds[] = {
        {0.4, 120, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 0., 0., 0., 0., 1.}
    };

    int result;
    result = pos3d_in_bounds(inside, bounds);

    ck_assert_msg( result == 1,
             "Was expecting True but found %f \n", result);

    result = pos3d_in_bounds(outside, bounds);

    ck_assert_msg( result == 0,
            "Was expecting False but found %f \n", result);

}
END_TEST

START_TEST(test_angle_acc)
{
    vec3d start = {0.0, 0.0, 0.0};
    vec3d pred  = {1.0, 1.0, 1.0};
    vec3d cand  = {1.1, 1.0, 1.0};

    double angle, acc;

    angle_acc(start, pred, cand, &angle, &acc);
    ck_assert_msg( fabs(angle - 2.902234) < EPS,
             "Was expecting 2.902234 but found %f \n", angle);

    ck_assert_msg( fabs(acc - 0.1) < EPS,
             "Was expecting 0.1 but found %f \n", acc);


    angle_acc(start, pred, pred, &angle, &acc);
    ck_assert_msg( fabs(acc) < EPS,
                      "Was expecting 0.0 but found %f \n", acc);
    ck_assert_msg( fabs(angle) < EPS,
                 "Was expecting 0.0 but found %f \n", angle);

    vec_scalar_mul(pred,-1,cand);
    angle_acc(start, pred, cand, &angle, &acc);

     ck_assert_msg( fabs(angle - 200.0) < EPS,
                    "Was expecting 200.0 but found %f \n", angle);

}
END_TEST




START_TEST(test_candsearch_in_pix)
{
    double cent_x, cent_y, dl, dr, du, dd;
    int p[4], counter = 0;

    target test_pix[] = {
        {0, 0.0, -0.2, 5, 1, 2, 10, -999},
        {6, 0.2, 0.2, 10, 8, 1, 20, -999},
        {3, 0.2, 0.3, 10, 3, 3, 30, -999},
        {4, 0.2, 1.0, 10, 3, 3, 40, -999},
        {1, -0.7, 1.2, 10, 3, 3, 50, -999},
        {7, 1.2, 1.3, 10, 3, 3, 60, -999},
        {5, 10.4, 2.1, 10, 3, 3, 70, -999}
    };
    int num_targets = 7;


    /* prepare test control parameters, basically for pix_x  */
    int cam;
    char img_format[] = "cam%d";
    char cal_format[] = "cal/cam%d.tif";
    control_par *test_cpar;

    test_cpar = new_control_par(4);
    for (cam = 0; cam < 4; cam++) {
        sprintf(test_cpar->img_base_name[cam], img_format, cam + 1);
        sprintf(test_cpar->cal_img_base_name[cam], cal_format, cam + 1);
    }
    test_cpar->hp_flag = 1;
    test_cpar->allCam_flag = 0;
    test_cpar->tiff_flag = 1;
    test_cpar->imx = 1280;
    test_cpar->imy = 1024;
    test_cpar->pix_x = 0.02; /* 20 micron pixel */
    test_cpar->pix_y = 0.02;
    test_cpar->chfield = 0;
    test_cpar->mm->n1 = 1;
    test_cpar->mm->n2[0] = 1.49;
    test_cpar->mm->n3 = 1.33;
    test_cpar->mm->d[0] = 5;

    cent_x = cent_y = 0.2;
    dl = dr = du = dd = 0.1;

    counter = candsearch_in_pix (test_pix, num_targets, cent_x, cent_y, \
                                 dl, dr, du, dd, p, test_cpar);

    printf("counter %d \n",counter);
    printf("candidates: \n");
    for (int i=0;i<counter;i++){
        printf("%f,%f\n",test_pix[p[i]].x,test_pix[p[i]].y);
    }
    fail_unless(counter == 2);


    cent_x = 0.5;
    cent_y = 0.3;
    dl = dr = du = dd = 10.2;

    counter = candsearch_in_pix (test_pix, num_targets, cent_x, cent_y, \
                                 dl, dr, du, dd, p, test_cpar);
    printf("counter %d \n",counter);
    printf("candidates:\n");
    for (int i=0;i<counter;i++){
        printf("%f,%f\n",test_pix[p[i]].x,test_pix[p[i]].y);
    }

    fail_unless(counter == 4);

}
END_TEST

START_TEST(test_sort)
{
    float test_array[] = {1.0, 2200.2, 0.3, -0.8, 100.0};
    int ix_array[] = {0,5,13,2,124};
    int len_array = 5;

    sort(len_array,test_array,ix_array);

    ck_assert_msg( fabs(test_array[0] + 0.8) < EPS,
             "Was expecting -0.8 but found %f \n", fabs(test_array[0]));

    ck_assert_msg( ix_array[len_array-1] != 1,
             "Was expecting 1 but found %f \n", ix_array[len_array-1]);

    printf("Sorted array:\n");
    for (int i=0;i<len_array;i++){
        printf("test_array[%d]=%f\n",ix_array[i],test_array[i]);
    }

}
END_TEST

START_TEST(test_copy_foundpix_array)
{
    foundpix src[] = {  {1,1,{1,0}},
                        {2,5,{1,1}}
                    };
    foundpix *dest;
    int arr_len = 2;
    int num_cams = 2;

    dest = (foundpix *) calloc (arr_len, sizeof (foundpix));

    reset_foundpix_array(dest, arr_len, num_cams);
    ck_assert_msg( dest[1].ftnr == -1 ,
             "Was expecting dest[1].ftnr == -1 but found %d \n", dest[1].ftnr);
    ck_assert_msg( dest[0].freq == 0 ,
              "Was expecting dest.freq == 0 but found %d \n", dest[0].freq);
    ck_assert_msg( dest[1].whichcam[0] == 0 ,
                       "Was expecting 0 but found %d \n", dest[1].whichcam[0]);



    copy_foundpix_array(dest, src, 2, 2);

    ck_assert_msg( dest[1].ftnr == 2 ,
             "Was expecting dest[1].ftnr == 2 but found %d \n", dest[1].ftnr);

    printf(" destination foundpix array\n");
    for (int i=0; i<arr_len; i++){
        printf("ftnr = %d freq=%d whichcam = %d %d\n", dest[i].ftnr, dest[i].freq, \
        dest[i].whichcam[0],dest[i].whichcam[1]);
    }


}
END_TEST


START_TEST(test_searchquader)
{
    vec3d point = {185.5, 3.2, 203.9};
    double xr[4], xl[4], yd[4], yu[4];
    Calibration *calib[3];
    control_par *cpar;
    
    chdir("testing_fodder/track");

    fail_if((cpar = read_control_par("parameters/ptv.par"))== 0);
    cpar->mm->n2[0] = 1.0000001;
    cpar->mm->n3 = 1.0000001;

    track_par tpar[] = {
        {0.4, 120, 0.2, -0.2, 0.1, -0.1, 0.1, -0.1, 0., 0., 0., 0., 1.}
    };

    read_all_calibration(calib, cpar->num_cams);

    searchquader(point, xr, xl, yd, yu, tpar, cpar, calib);

    //printf("searchquader returned:\n");
    //for (int i=0; i<cpar->num_cams;i++){
    //     printf("%f %f %f %f\n",xr[i],xl[i],yd[i],yu[i]);
    // }
    
    ck_assert_msg( fabs(xr[0] - 0.560048)<EPS ,
             "Was expecting 0.560048 but found %f \n", xr[0]);
    ck_assert_msg( fabs(yu[1] - 0.437303)<EPS ,
                      "Was expecting 0.437303 but found %f \n", yu[1]);
    
    /* let's test just one camera, if there are no problems with the borders */
    
    cpar->num_cams = 1;
    track_par tpar1[] = {
        {0.4, 120, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0., 0., 0., 0., 1.}
    };
    searchquader(point, xr, xl, yd, yu, tpar1, cpar, calib);
    ck_assert_msg( fabs(xr[0] - 0.0)<EPS ,
                  "Was expecting 0.0 but found %f \n", xr[0]);
    
    /* test infinitely large values of tpar that should return about half the image size */
    track_par tpar2[] = {
        {0.4, 120, 1000.0, -1000.0, 1000.0, -1000.0, 1000.0, -1000.0, 0., 0., 0., 0., 1.}
    };
    searchquader(point, xr, xl, yd, yu, tpar2, cpar, calib);
    ck_assert_msg( fabs(xr[0] + xl[0] - cpar->imx)<EPS ,
                  "Was expecting image size but found %f \n", xr[0]+xl[0]);
    ck_assert_msg( fabs(yd[0] + yu[0] - cpar->imy)<EPS ,
                  "Was expecting cpar->imy but found %f \n", yd[0]+yu[0]);


}
END_TEST


START_TEST(test_sortwhatfound)
{
    foundpix src[] = {  {1,0,{1,0}},
                        {2,0,{1,1}}
                     };
    foundpix *dest;
    int num_cams = 2;
    
    /* sortwhatfound freaks out if array is not reset before */
    dest = (foundpix *) calloc (num_cams*MAX_CANDS, sizeof (foundpix));
    reset_foundpix_array(dest, num_cams*MAX_CANDS, num_cams);
    copy_foundpix_array(dest, src, 2, 2);

    
    int counter;

    /* test simple sort of a small foundpix array */
    sortwhatfound(dest, &counter, num_cams);
    
    ck_assert_msg( dest[0].ftnr == 2 ,
                  "Was expecting dest[0].ftnr == 2 but found %d \n", dest[0].ftnr);
    ck_assert_msg( dest[0].freq == 2 ,
                  "Was expecting dest[0].freq == 2 but found %d \n", dest[0].freq);
    ck_assert_msg( dest[1].freq == 0 ,
                  "Was expecting dest[1].freq == 0 but found %d \n", dest[1].freq);

}
END_TEST

START_TEST(test_trackcorr_c_loop)
{
    tracking_run *ret;
    int step, display=0;
    Calibration *calib[3];
    control_par *cpar;

    chdir("testing_fodder/track");
    
    printf("----------------------------\n");
    printf("Test tracking multiple files 2 cameras, 1 particle \n");
    
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    ret = trackcorr_c_init(calib[0]);


    trackcorr_c_loop (ret, ret->seq_par->first, display, calib);
    
    for (step = ret->seq_par->first+1; step < ret->seq_par->last; step++)
    {
        trackcorr_c_loop (ret, step, display, calib);
    }
    trackcorr_c_finish(ret, ret->seq_par->last, display);
    
    int range = ret->seq_par->last - ret->seq_par->first;
    double npart, nlinks;
    
    /* average of all steps */
    npart = (double)ret->npart / range;
    nlinks = (double)ret->nlinks / range;
    
    ck_assert_msg(fabs(npart - 208.0/210.0)<EPS,
                  "Was expecting npart == 208/210 but found %f \n", npart);
    ck_assert_msg(fabs(nlinks - 198.0/210.0)<EPS,
                  "Was expecting nlinks == 198/210 but found %f \n", nlinks);
    
}
END_TEST

START_TEST(test_cavity)
{
    tracking_run *ret;
    int display=0;
    Calibration *calib[4];
    control_par *cpar;
    
    
    printf("----------------------------\n");
    printf("Test cavity case \n");
    
    chdir("testing_fodder/test_cavity");
    
    
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    printf("In test_cavity num cams = %d\n",cpar->num_cams);
    ret = trackcorr_c_init(calib[0]);
    
    trackcorr_c_loop (ret, 10002, display, calib);
    //trackcorr_c_finish(ret, 10002, display);
    
    ck_assert_msg(ret->npart == 672,
                  "Was expecting npart == 672 but found %f \n", ret->npart);
    ck_assert_msg(ret->nlinks == 99,
                  "Was expecting nlinks == 99 but found %f \n", ret->nlinks);
}
END_TEST

START_TEST(test_trackback)
{
    tracking_run *ret;
    int display=0;
    double nlinks;
    Calibration *calib[3];
    control_par *cpar;
    
    chdir("testing_fodder/track");
    
    printf("----------------------------\n");
    printf("trackback test \n");
    
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    ret = trackcorr_c_init(calib[0]);
    ret->tpar->dvxmin =ret->tpar->dvymin=ret->tpar->dvzmin=-50;
    ret->tpar->dvxmax =ret->tpar->dvymax=ret->tpar->dvzmax=50;
    
    ret->lmax = norm((ret->tpar->dvxmin - ret->tpar->dvxmax), \
                     (ret->tpar->dvymin - ret->tpar->dvymax), \
                     (ret->tpar->dvzmin - ret->tpar->dvzmax));
    
    nlinks = trackback_c(ret, ret->seq_par->last, display, calib);
    
    ck_assert_msg(fabs(nlinks - 201.0/209.0)<EPS,
                  "Was expecting nlinks to be 201/209 but found %f %f\n", nlinks, nlinks*209.0);
    
    
    
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("ttools");

    TCase *tc = tcase_create ("predict test");
    tcase_add_test(tc, test_predict);
    suite_add_tcase (s, tc);

    tc = tcase_create ("search_volume_center_moving");
    tcase_add_test(tc, test_search_volume_center_moving);
    suite_add_tcase (s, tc);

    tc = tcase_create ("pos3d_in_bounds");
    tcase_add_test(tc, test_pos3d_in_bounds);
    suite_add_tcase (s, tc);

    tc = tcase_create ("angle_acc");
    tcase_add_test(tc, test_angle_acc);
    suite_add_tcase (s, tc);

    tc = tcase_create ("candsearch_in_pix");
    tcase_add_test(tc, test_candsearch_in_pix);
    suite_add_tcase (s, tc);

    tc = tcase_create ("sort");
    tcase_add_test(tc, test_sort);
    suite_add_tcase (s, tc);

    tc = tcase_create ("reset_copy_foundpix_array");
    tcase_add_test(tc, test_copy_foundpix_array);
    suite_add_tcase (s, tc);

    tc = tcase_create ("searchquader");
    tcase_add_test(tc, test_searchquader);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Sortwhatfound");
    tcase_add_test(tc, test_sortwhatfound);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Test cavity case");
    tcase_add_test(tc, test_cavity);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Trackcorr_c_loop");
    tcase_add_test(tc, test_trackcorr_c_loop);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Trackback");
    tcase_add_test(tc, test_trackback);
    suite_add_tcase (s, tc);


    return s;
}

int main(void) {
    int number_failed;
    Suite *s = fb_suite ();
    SRunner *sr = srunner_create (s);
    // srunner_run_all (sr, CK_ENV);
    srunner_run_all (sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
