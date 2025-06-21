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

START_TEST(test_trackcorr_no_add)
{
    // ...existing code...

    // Replace trackcorr_c_loop with track3d_loop
    track3d_loop(run, step);

    // ...existing code...
}
END_TEST

START_TEST(test_trackcorr_with_add)
{
    // ...existing code...

    // Replace trackcorr_c_loop with track3d_loop
    track3d_loop(run, step);

    // ...existing code...
}
END_TEST

START_TEST(test_cavity)
{
    // ...existing code...

    // Replace trackcorr_c_loop with track3d_loop
    track3d_loop(run, step);

    // ...existing code...
}
END_TEST

START_TEST(test_burgers)
{
    // ...existing code...

    // Replace trackcorr_c_loop with track3d_loop
    track3d_loop(run, step);

    // ...existing code...
}
END_TEST

START_TEST(test_new_particle)
{
    // ...existing code...

    // Replace trackcorr_c_loop with track3d_loop
    track3d_loop(run, step);

    // ...existing code...
}
END_TEST

// --- Suite and main (copied from check_track.c) ---

Suite *track3d_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Track3D");

    /* Core test case */
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_trackcorr_no_add);
    tcase_add_test(tc_core, test_trackcorr_with_add);
    tcase_add_test(tc_core, test_cavity);
    tcase_add_test(tc_core, test_burgers);
    tcase_add_test(tc_core, test_new_particle);
    // ...add other test cases as needed...

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

    srunner_set_fork_status(sr, CK_NOFORK);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
