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
#include <dirent.h>
#include <math.h>
#include "track.h"
#include "calibration.h"

#define EPS 1E-5

void read_all_calibration(Calibration *calib[], int num_cams) {
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

int copy_res_dir(char *src, char *dest) {
    DIR *dirp;
    struct dirent *dp;
    FILE *in_f, *out_f;
    int errno;
    char file_name[256];
    char buf[8192];
    ssize_t result;

    dirp = opendir(src);
    while (dirp) {
        errno = 0;
        if ((dp = readdir(dirp)) != NULL) {
            if (dp->d_name[0] == '.') continue;

            strncpy(file_name, src, 255);
            strncat(file_name, dp->d_name, 255);
            in_f = fopen(file_name, "r");
            strncpy(file_name, dest, 255);
            strncat(file_name, dp->d_name, 255);
            out_f = fopen(file_name, "w");
            
            while (!feof(in_f)) {
                result = fread(buf, 1, sizeof(buf), in_f);
                fwrite(buf, 1, result, out_f);
            }
            fclose(in_f);
            fclose(out_f);
        } else {
            closedir(dirp);
            return 1;
        }
    }
}

int empty_res_dir() {
    DIR *dirp;
    struct dirent *dp;
    int errno;
    char file_name[256];
    ssize_t result;

    dirp = opendir("res/");
    while (dirp) {
        errno = 0;
        if ((dp = readdir(dirp)) != NULL) {
            if (dp->d_name[0] == '.') continue;
            strncpy(file_name, "res/", 255);
            strncat(file_name, dp->d_name, 255);
            remove(file_name);
        } else {
            closedir(dirp);
            return 1;
        }
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
    int cam, i;
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
    for (i=0;i<counter;i++){
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
    for (i=0;i<counter;i++){
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
    int i;

    sort(len_array,test_array,ix_array);

    ck_assert_msg( fabs(test_array[0] + 0.8) < EPS,
             "Was expecting -0.8 but found %f \n", fabs(test_array[0]));

    ck_assert_msg( ix_array[len_array-1] != 1,
             "Was expecting 1 but found %f \n", ix_array[len_array-1]);

    printf("Sorted array:\n");
    for (i=0;i<len_array;i++){
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
    int i;

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
    for (i=0; i<arr_len; i++){
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


START_TEST(test_sort_candidates_by_freq)
{
    foundpix src[] = {  {1,0,{1,0}},
                        {2,0,{1,1}}
                     };
    foundpix *dest;
    int num_cams = 2;
    int num_parts;
    
    /* sortwhatfound freaks out if array is not reset before */
    dest = (foundpix *) calloc (num_cams*MAX_CANDS, sizeof (foundpix));
    reset_foundpix_array(dest, num_cams*MAX_CANDS, num_cams);
    copy_foundpix_array(dest, src, 2, 2);

    /* test simple sort of a small foundpix array */
    num_parts = sort_candidates_by_freq(dest, num_cams);
    
    ck_assert_msg( dest[0].ftnr == 2 ,
                  "Was expecting dest[0].ftnr == 2 but found %d \n", dest[0].ftnr);
    ck_assert_msg( dest[0].freq == 2 ,
                  "Was expecting dest[0].freq == 2 but found %d \n", dest[0].freq);
    ck_assert_msg( dest[1].freq == 0 ,
                  "Was expecting dest[1].freq == 0 but found %d \n", dest[1].freq);

}
END_TEST

START_TEST(test_trackcorr_no_add)
{
    tracking_run *run;
    int step;
    Calibration *calib[3];
    control_par *cpar;

    chdir("testing_fodder/track");
    copy_res_dir("res_orig/", "res/");
    copy_res_dir("img_orig/", "img/");
    
    printf("----------------------------\n");
    printf("Test tracking multiple files 2 cameras, 1 particle \n");
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    
    run = tr_new_legacy("parameters/sequence.par", 
        "parameters/track.par", "parameters/criteria.par", 
        "parameters/ptv.par", calib);
    run->tpar->add = 0;
    track_forward_start(run);
    trackcorr_c_loop(run, run->seq_par->first);
    
    for (step = run->seq_par->first + 1; step < run->seq_par->last; step++) {
        trackcorr_c_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);
    empty_res_dir();
    
    int range = run->seq_par->last - run->seq_par->first;
    double npart, nlinks;
    
    /* average of all steps */
    npart = (double)run->npart / range;
    nlinks = (double)run->nlinks / range;
    
    ck_assert_msg(fabs(npart - 206.0/210.0)<EPS,
                  "Was expecting npart == 208/210 but found %f \n", npart);
    ck_assert_msg(fabs(nlinks - 198.0/210.0)<EPS,
                  "Was expecting nlinks == 198/210 but found %f \n", nlinks);
    
}
END_TEST

START_TEST(test_trackcorr_with_add)
{
    tracking_run *run;
    int step;
    Calibration *calib[3];
    control_par *cpar;

    chdir("testing_fodder/track");
    copy_res_dir("res_orig/", "res/");
    copy_res_dir("img_orig/", "img/");
    
    printf("----------------------------\n");
    printf("Test tracking multiple files 2 cameras, 1 particle \n");
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    
    run = tr_new_legacy("parameters/sequence.par", 
        "parameters/track.par", "parameters/criteria.par", 
        "parameters/ptv.par", calib);
    track_forward_start(run);
    trackcorr_c_loop(run, run->seq_par->first);
    
    for (step = run->seq_par->first + 1; step < run->seq_par->last; step++) {
        trackcorr_c_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);
    empty_res_dir();
    
    int range = run->seq_par->last - run->seq_par->first;
    double npart, nlinks;
    
    /* average of all steps */
    npart = (double)run->npart / range;
    nlinks = (double)run->nlinks / range;
    
    ck_assert_msg(fabs(npart - 1928.0/210.0)<EPS,
                  "Was expecting npart == 208/210 but found %f \n", npart);
    ck_assert_msg(fabs(nlinks - 328.0/210.0)<EPS,
                  "Was expecting nlinks == 328/210 but found %f \n", nlinks);
    
}
END_TEST

START_TEST(test_cavity)
{
    tracking_run *ret;
    Calibration *calib[4];
    control_par *cpar;
    
    
    printf("----------------------------\n");
    printf("Test cavity case \n");
    
    chdir("testing_fodder/test_cavity");
    copy_res_dir("res_orig/", "res/");
    copy_res_dir("img_orig/", "img/");
    
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    printf("In test_cavity num cams = %d\n",cpar->num_cams);
    ret = tr_new_legacy("parameters/sequence.par", 
        "parameters/track.par", "parameters/criteria.par", 
        "parameters/ptv.par", calib);
    track_forward_start(ret);
    
    trackcorr_c_loop (ret, 10002);
    empty_res_dir();
    
    ck_assert_msg(ret->npart == 672,
                  "Was expecting npart == 672 but found %d \n", ret->npart);
    ck_assert_msg(ret->nlinks == 127,
                  "Was expecting nlinks == 127 but found %d \n", ret->nlinks);
    
    trackcorr_c_finish(ret, 10002);
}
END_TEST

START_TEST(test_trackback)
{
    tracking_run *run;
    double nlinks;
    int step;
    Calibration *calib[3];
    control_par *cpar;
    
    chdir("testing_fodder/track");
    copy_res_dir("res_orig/", "res/");
    copy_res_dir("img_orig/", "img/");
    
    printf("----------------------------\n");
    printf("trackback test \n");
    
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);
    run = tr_new_legacy("parameters/sequence.par",
        "parameters/track.par", "parameters/criteria.par",
        "parameters/ptv.par", calib);
    track_forward_start(run);
    trackcorr_c_loop(run, run->seq_par->first);
    
    for (step = run->seq_par->first + 1; step < run->seq_par->last; step++) {
        trackcorr_c_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);
    run->tpar->dvxmin = run->tpar->dvymin = run->tpar->dvzmin = -50;
    run->tpar->dvxmax = run->tpar->dvymax = run->tpar->dvzmax = 50;
    run->lmax = norm((run->tpar->dvxmin - run->tpar->dvxmax), \
                     (run->tpar->dvymin - run->tpar->dvymax), \
                     (run->tpar->dvzmin - run->tpar->dvzmax));
    
    nlinks = trackback_c(run, run->seq_par->last);
    empty_res_dir();
    
    ck_assert_msg(fabs(nlinks - 1.043062)<EPS,
                  "Was expecting nlinks to be 1.043062 but found %f\n", nlinks);
}
END_TEST

START_TEST(test_new_particle)
{
    /* this test also has the side-effect of testing instantiation of a 
       tracking_run struct without the legacy stuff. */

    Calibration *calib[3];
    control_par *cpar;
    sequence_par *spar;
    track_par *tpar;
    volume_par *vpar;
    tracking_run *run;
    
    char ori_tmpl[] = "cal/sym_cam%d.tif.ori";
    char added_name[] = "cal/cam1.tif.addpar";
    char ori_name[256];
    int cam, status;

    fail_unless((status = chdir("testing_fodder/")) == 0);
    
    /* Set up all scene parameters to track one specially-contrived 
       trajectory. */
    for (cam = 0; cam < 3; cam++) {
        sprintf(ori_name, ori_tmpl, cam + 1);
        calib[cam] = read_calibration(ori_name, added_name, NULL);
    }
    
    fail_unless((status = chdir("track/")) == 0);
    copy_res_dir("res_orig/", "res/");
    copy_res_dir("img_orig/", "img/");

    spar = read_sequence_par("parameters/sequence_newpart.par", 3);
    cpar = read_control_par("parameters/control_newpart.par");
    tpar = read_track_par("parameters/track.par");
    vpar = read_volume_par("parameters/criteria.par");
    
    run = tr_new(spar, tpar, vpar, cpar, 4, MAX_TARGETS, 
        "res/particles", "res/linkage", "res/whatever", calib, 0.1);
    track_forward_start(run);
    trackcorr_c_loop(run, 1);
    trackcorr_c_loop(run, 2);
    
    fb_prev(run->fb); /* because each loop step moves the FB forward */
    fail_unless(run->fb->buf[1]->path_info[0].next == 1);
    
    tpar->add = 0;
    track_forward_start(run);
    trackcorr_c_loop(run, 1);
    trackcorr_c_loop(run, 2);
    fb_prev(run->fb); /* because each loop step moves the FB forward */

    empty_res_dir();
    
    fail_unless(run->fb->buf[1]->path_info[0].next == 0);
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
    
    tc = tcase_create ("sort_candidates_by_freq");
    tcase_add_test(tc, test_sort_candidates_by_freq);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Test cavity case");
    tcase_add_test(tc, test_cavity);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Tracking forward without additions");
    tcase_add_test(tc, test_trackcorr_no_add);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Tracking forward with adding particles");
    tcase_add_test(tc, test_trackcorr_with_add);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Trackback");
    tcase_add_test(tc, test_trackback);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Tracking a constructed frame");
    tcase_add_test(tc, test_new_particle);
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
