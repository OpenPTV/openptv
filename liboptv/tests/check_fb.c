/*  Unit tests for functions related to the frame buffer. Uses the Check
    framework: http://check.sourceforge.net/
    
    To run it, type "make check" when in the top C directory, src_c/
    If that doesn't run the tests, use the Check tutorial:
    http://check.sourceforge.net/doc/check_html/check_3.html
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>

#include "tracking_frame_buf.h"

START_TEST(test_read_targets)
{
    target tbuf[2]; /* Two targets in the sample target file */
    target t1 = {0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1};
    target t2 = {1, 796.0000, 809.0000, 13108, 113, 116, 658928, 0};
    
    char *file_base = "testing_fodder/sample_";
    int frame_num = 42;
    int targets_read = 0;
    
    targets_read = read_targets(tbuf, file_base, frame_num);
    fail_unless(targets_read == 2);
    fail_unless(compare_targets(tbuf, &t1));
    fail_unless(compare_targets(tbuf + 1, &t2));
}
END_TEST

START_TEST(test_zero_targets)
{
    /* zero targets should not generate an error value, but just return 0 */
    target tbuf[2]; /* There is extra space, of course. */
    char *file_base = "testing_fodder/sample_";
    int frame_num = 1;
    int targets_read = 0;

    targets_read = read_targets(tbuf, file_base, frame_num);
    fail_unless(targets_read == 0);

}
END_TEST

START_TEST(test_write_targets)
{
    /* Write and read back two targets, make sure they're the same.
    Assumes that read_targets is ok, which is tested separately. */
    
    target tbuf[2]; /* Two targets in the sample target file */
    target t1 = {0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1};
    target t2 = {1, 796.0000, 809.0000, 13108, 113, 116, 658928, 0};
    
    char *file_base = "testing_fodder/test_";
    int frame_num = 42;
    int num_targets = 2;
    
    tbuf[0] = t1; tbuf[1] = t2;
    fail_unless(write_targets(tbuf, num_targets, file_base, frame_num));
    
    num_targets = read_targets(tbuf, file_base, frame_num);
    fail_unless(num_targets == 2);
    fail_unless(compare_targets(tbuf, &t1));
    fail_unless(compare_targets(tbuf + 1, &t2));
    
    // Leave the test directory clean:
    remove("testing_fodder/test_0042_targets");
}
END_TEST

START_TEST(test_read_path_frame)
{
    corres cor_buf[80];
    P path_buf[80];
    int alt_link;
    
    /* Correct values for particle 3 */
    P path_correct = {
        .x = {45.219, -20.269, 25.946},
        .prev = -1,
        .next = -2,
        .prio = 4, 
        .finaldecis = 1000000.0,
        .inlist = 0.
    };
    for (alt_link = 0; alt_link < POSI; alt_link++) {
        path_correct.decis[alt_link] = 0.0;
        path_correct.linkdecis[alt_link] = -999;
    }
    corres c_correct = { 3, {96, 66, 26, 26} };

    char *file_base = "testing_fodder/rt_is";
    int frame_num = 818;
    int targets_read = 0;
    
    /* Test unlinked frame: */ 
    targets_read = read_path_frame(cor_buf, path_buf, file_base, NULL, NULL,
        frame_num);
    fail_unless(targets_read == 80);
    
    fail_unless(compare_corres(cor_buf + 2, &c_correct), \
        "Got corres: %d, [%d %d %d %d]", cor_buf[2].nr, \
        cor_buf[2].p[0], cor_buf[2].p[1], cor_buf[2].p[2], cor_buf[2].p[3]);
    fail_unless(compare_path_info(path_buf + 2, &path_correct));
    
    /* Test frame with links */
    path_correct.prev = 0;
    path_correct.next = 0;
    path_correct.prio = 0;
    char *linkage_base = "testing_fodder/ptv_is";
    char *prio_base = "testing_fodder/added";
    
    targets_read = read_path_frame(cor_buf, path_buf, file_base, linkage_base,
        prio_base, frame_num);
    fail_unless(targets_read == 80);
    fail_unless(compare_corres(cor_buf + 2, &c_correct), \
        "Got corres: %d, [%d %d %d %d]", cor_buf[2].nr, \
        cor_buf[2].p[0], cor_buf[2].p[1], cor_buf[2].p[2], cor_buf[2].p[3]);
    fail_unless(compare_path_info(path_buf + 2, &path_correct));
}
END_TEST

START_TEST(test_write_path_frame)
{
    corres cor_buf[] = { {1, {96, 66, 26, 26}}, {2, {30, 31, 32, 33}} };
    P path_buf[] = {
        {
        .x = {45.219, -20.269, 25.946},
        .prev = -1,
        .next = -2,
        .prio = 4,
        .finaldecis = 1000000.0,
        .inlist = 0.
        },
        {
        .x = {45.219, -20.269, 25.946},
        .prev = -1,
        .next = -2,
        .prio = 0,
        .finaldecis = 2000000.0,
        .inlist = 1.
        }
    };
    
    char *corres_file_base = "testing_fodder/rt_is";
    char *linkage_file_base = "testing_fodder/ptv_is";
    int frame_num = 42;
    
    fail_unless(write_path_frame(cor_buf, path_buf, 2,\
        corres_file_base, linkage_file_base, NULL, frame_num));
    
    remove("testing_fodder/rt_is.42");
    remove("testing_fodder/ptv_is.42");
}
END_TEST

START_TEST(test_init_frame)
{
    frame frm;
    
    // Dummy things to store in the frame's buffers:
    target t_target;
    corres t_corres;
    P t_path;
    
    int cams = 4;
    int max_targets = 100;
    int cam_ix = 0;
    
    frame_init(&frm, cams, max_targets);
    
    /* Try to write stuff into the allocated memory and see it doesn't
    segfault.*/
    frm.correspond[42] = t_corres;
    frm.path_info[42] = t_path;
    
    for (cam_ix = 0; cam_ix < cams; cam_ix ++) {
        frm.targets[cam_ix][42] = t_target;
    }
    
    fail_unless(frm.num_cams == cams);
    fail_unless(frm.max_targets == max_targets);
}
END_TEST

/* *_frame() are just simple wrappers around *_path_frame and *_targets, so
   we only do a simple write-read cycle. The heavy testing is in the wrapped
   functions. */
START_TEST(test_read_write_frame)
{
    frame frm, readback;
    char* target_files[2] = {
        "testing_fodder/target_test_cam0", "testing_fodder/target_test_cam1"};
    char *corres_base = "testing_fodder/corres_test";
    char *linkage_base = "testing_fodder/ptv_test";
    char *prio_base = "testing_fodder/added_test";
    char namebuf[100]; /* For removals */
    int alt_link;
    
    // Dummy things to store in the frame's buffers:
    target t_target = {0, 1127.0000, 796.0000, 13320, 111, 120, 828903, 1};
    corres t_corres = { 3, {96, 66, 26, 26} };
    P t_path = {
        .x = {45.219, -20.269, 25.946},
        .prev = -1,
        .next = -2,
        .prio = 4,
        .finaldecis = 1000000.0,
        .inlist = 0.
    };
    for (alt_link = 0; alt_link < POSI; alt_link++) {
        t_path.decis[alt_link] = 0.0;
        t_path.linkdecis[alt_link] = -999;
    }
    
    int cams = 2;
    int max_targets = 100;
    int cam_ix = 0;
    int frame_num = 7;
    
    frame_init(&frm, cams, max_targets);
    
    /* Try to write stuff into the allocated memory and see it doesn't
    segfault.*/
    frm.correspond[2] = t_corres;
    frm.path_info[2] = t_path;
    frm.num_parts = 3;
    
    for (cam_ix = 0; cam_ix < cams; cam_ix++) {
        frm.targets[cam_ix][42] = t_target;
        frm.num_targets[cam_ix] = 43;
    }
    /* Zero out the second camera targets, should not complain */
    frm.num_targets[cams - 1] = 0;
    
    fail_unless(write_frame(&frm, corres_base, linkage_base, NULL,
        target_files, frame_num));
    
    frame_init(&readback, cams, max_targets);
    fail_unless(read_frame(&readback, corres_base, NULL, NULL, target_files,
        frame_num));
    
    fail_unless(compare_corres(&t_corres, readback.correspond + 2));
    fail_unless(compare_path_info(&t_path, readback.path_info + 2));
    fail_unless(compare_targets(&t_target, readback.targets[0] + 42));
    
    /* Now read frame with links: */
    t_path.prev = 0;
    t_path.next = 0;
    t_path.prio = 0;
    frm.path_info[2] = t_path;
    
    fail_unless(write_frame(&frm, corres_base, linkage_base, prio_base,
        target_files, frame_num));
    
    frame_init(&readback, cams, max_targets);
    fail_unless(read_frame(&readback, corres_base, linkage_base, prio_base,
        target_files, frame_num));
    
    fail_unless(compare_corres(&t_corres, readback.correspond + 2));
    fail_unless(compare_path_info(&t_path, readback.path_info + 2));
    fail_unless(compare_targets(&t_target, readback.targets[0] + 42));
    
    sprintf(namebuf, "%s.%d", corres_base, frame_num);
    remove(namebuf);
    sprintf(namebuf, "%s.%d", linkage_base, frame_num);
    remove(namebuf);
    sprintf(namebuf, "%s.%d", prio_base, frame_num);
    remove(namebuf);
    remove("testing_fodder/target_test_cam10007_targets");
    remove("testing_fodder/target_test_cam00007_targets");
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("Frame Buffer");

    TCase *tc_trt = tcase_create ("Read targets");
    tcase_add_test(tc_trt, test_read_targets);
    suite_add_tcase (s, tc_trt);

    tc_trt = tcase_create ("Read zero targets");
    tcase_add_test(tc_trt, test_zero_targets);
    suite_add_tcase (s, tc_trt);
    
    TCase *tc_twt = tcase_create ("Write targets");
    tcase_add_test(tc_twt, test_write_targets);
    suite_add_tcase (s, tc_twt);

    TCase *tc_trpf = tcase_create ("Read path frame");
    tcase_add_test(tc_trpf, test_read_path_frame);
    suite_add_tcase (s, tc_trpf);

    TCase *tc_twpf = tcase_create ("Write path frame");
    tcase_add_test(tc_twpf, test_write_path_frame);
    suite_add_tcase (s, tc_twpf);

    TCase *tc_tcf = tcase_create ("Create frame");
    tcase_add_test(tc_tcf, test_init_frame);
    suite_add_tcase (s, tc_tcf);
    
    TCase *tc_trwf = tcase_create ("Write/read frame");
    tcase_add_test(tc_trwf, test_read_write_frame);
    suite_add_tcase (s, tc_trwf);

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

