/*  Unit tests for the 3D tracking (track3d_loop). Uses the Check framework.
    This is a copy of check_track.c, but calls track3d_loop instead of trackcorr_c_loop.
    Copyright 2025 OpenPTV contributors
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <math.h>
#include "track.h"
#include "track3d.h"
#include "calibration.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>

#define EPS 1E-5

// Helper functions
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
            file_name[255] = '\0'; // Ensure null termination
            strncat(file_name, dp->d_name, 255 - strlen(file_name) - 1); // Adjust size for remaining space
            in_f = fopen(file_name, "r");
            strncpy(file_name, dest, 255);
            file_name[255] = '\0'; // Ensure null termination
            strncat(file_name, dp->d_name, 255 - strlen(file_name) - 1); // Adjust size for remaining space
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
    return 0;
}

int empty_res_dir() {
    DIR *dirp;
    struct dirent *dp;
    int errno;
    char file_name[256];
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
    return 0;
}

START_TEST(test_track3d_no_add)
{
     tracking_run *run;
    int step;
    Calibration *calib[3];
    control_par *cpar;
    char cwd[PATH_MAX];


    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
    }  
    

    chdir("testing_fodder/track");
    // chdir("testing_fodder/track");
    copy_res_dir("res_orig/", "res/");
    copy_res_dir("img_orig/", "img/");

    printf("----------------------------\n");
    printf("Test tracking multiple files 2 cameras, 1 particle\n");
    cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar->num_cams);

    run = tr_new_legacy("parameters/sequence.par",
        "parameters/track.par", "parameters/criteria.par",
        "parameters/ptv.par", calib);
    run->tpar->add = 0;



    track_forward_start(run);
    track3d_loop(run, run->seq_par->first);

    for (step = run->seq_par->first + 1; step < run->seq_par->last; step++) {
        track3d_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);

    empty_res_dir();
    
    int range = run->seq_par->last - run->seq_par->first;
    double npart, nlinks;
    
    /* average of all steps */
    npart = (double)run->npart / range;
    nlinks = (double)run->nlinks / range;

    printf("npart: %d\n", run->npart);
    printf("nlinks: %d\n", run->nlinks);
    
    ck_assert_msg(fabs(npart - 0.8)<EPS,
                  "Was expecting npart == 208/210 but found %f \n", npart);
    ck_assert_msg(fabs(nlinks - 0.8)<EPS,
                  "Was expecting nlinks == 198/210 but found %f \n", nlinks);
    

    // ...existing code...
}
END_TEST


START_TEST(track3d_test_cavity)
{
    tracking_run *run;
    Calibration *calib[4];
    control_par *cpar;
    int step;
    struct stat st = {0};
    char cwd[PATH_MAX];
    
    
    printf("----------------------------\n");
    printf("Test cavity case \n");

    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
    }  
    
    chdir("testing_fodder/test_cavity");  // after another test
    if (stat("res", &st) == -1) {
        mkdir("res", 0700);
    }
    copy_res_dir("res_orig/", "res/");
    
    if (stat("img", &st) == -1) {
        mkdir("img", 0700);
    }
    copy_res_dir("img_orig/", "img/");

    fail_if((cpar = read_control_par("parameters/ptv.par"))== 0);
    read_all_calibration(calib, cpar->num_cams);

    run = tr_new_legacy("parameters/sequence.par", 
        "parameters/track.par", "parameters/criteria.par", 
        "parameters/ptv.par", calib);

    printf("num cams in run is %d\n",run->cpar->num_cams);
    printf("add particle is %d\n",run->tpar->add);

    track_forward_start(run);    
    for (step = run->seq_par->first; step < run->seq_par->last; step++) {
        track3d_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);
    printf("total num parts is %d, num links is %d \n", run->npart, run->nlinks);

    ck_assert_msg(run->npart == 672+699+711,
                  "Was expecting npart == 2082 but found %d \n", run->npart);
    ck_assert_msg(run->nlinks >= 132+176+144,
                  "Was expecting nlinks >= 452 found %ld \n", run->nlinks);

    
    empty_res_dir();
}
END_TEST

START_TEST(track3d_test_burgers)
{
    tracking_run *run;
    Calibration *calib[4];
    control_par *cpar;
    int status, step;
    struct stat st = {0};
    char cwd[PATH_MAX];



    printf("----------------------------\n");
    printf("Test Burgers vortex case with track3d \n");


    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
    }


    chdir("testing_fodder/burgers");

    // chdir("testing_fodder/burgers");

    if (stat("res", &st) == -1) {
        mkdir("res", 0700);
    }
    copy_res_dir("res_orig/", "res/");

    if (stat("img", &st) == -1) {
        mkdir("img", 0700);
    }
    copy_res_dir("img_orig/", "img/");

    fail_if((cpar = read_control_par("parameters/ptv.par"))== 0);
    read_all_calibration(calib, cpar->num_cams);

    run = tr_new_legacy("parameters/sequence.par",
        "parameters/track.par", "parameters/criteria.par",
        "parameters/ptv.par", calib);

    printf("num cams in run is %d\n", run->cpar->num_cams);
    printf("add particle is %d\n", run->tpar->add);

    track_forward_start(run);
    for (step = run->seq_par->first; step < run->seq_par->last; step++) {
        track3d_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);
    printf("total num parts is %d, num links is %d \n", run->npart, run->nlinks);

    ck_assert_msg(run->npart == 19,
                  "Was expecting npart == 19 but found %d \n", run->npart);
    ck_assert_msg(run->nlinks == 18,
                  "Was expecting nlinks == 18 found %ld \n", run->nlinks);


    // run = tr_new_legacy("parameters/sequence.par",
    //     "parameters/track.par", "parameters/criteria.par",
    //     "parameters/ptv.par", calib);

    // run->tpar->add = 1;
    // printf("changed add particle to %d\n", run->tpar->add);

    // track_forward_start(run);
    // for (step = run->seq_par->first; step < run->seq_par->last; step++) {
    //     track3d_loop(run, step);
    // }
    // trackcorr_c_finish(run, run->seq_par->last);
    // printf("total num parts is %d, num links is %d \n", run->npart, run->nlinks);

    // // ck_assert_msg(run->npart == 20,
    // //               "Was expecting npart == 20 but found %d \n", run->npart);
    // // ck_assert_msg(run->nlinks ==20,
    // //               "Was expecting nlinks == 20 but found %d \n", run->nlinks);

    empty_res_dir();

}
END_TEST


Suite *track3d_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Track3D");

    /* Core test case */
    tc_core = tcase_create("core test");
    tcase_add_test(tc_core, test_track3d_no_add);
    suite_add_tcase(s, tc_core);

    tc_core = tcase_create("test_cavity");
    tcase_add_test(tc_core, track3d_test_cavity);
    suite_add_tcase(s, tc_core);

    tc_core = tcase_create("burgers test");
    tcase_add_test(tc_core, track3d_test_burgers);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = track3d_suite();
    sr = srunner_create(s);

    // srunner_set_fork_status(sr, CK_NOFORK);

    srunner_run_all(sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
