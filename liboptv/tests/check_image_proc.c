/*  Unit tests for functions related to image processing. Uses the Check
    framework: http://check.sourceforge.net/
    
    To run it, type "make verify" when in the build/ directory created for 
    CMake.
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>

#include "parameters.h"
#include "image_processing.h"

int images_equal(unsigned char *img1, unsigned char *img2, int w, int h) {
    int pix;
    
    for (pix = w; pix < w*h; pix++) {
        printf("%d == %d; ", img1[pix], img2[pix]);
        if (img1[pix] != img2[pix])
            return 0; }
    return 1;
}

START_TEST(test_general_filter)
{
    filter_t blur_filt = {{0, 0.2, 0}, {0.2, 0.2, 0.2}, {0, 0.2, 0}};
    unsigned char img[5][5] = {
        { 0,   0,   0,   0, 0},
        { 0, 255, 255, 255, 0},
        { 0, 255, 255, 255, 0},
        { 0, 255, 255, 255, 0},
        { 0,   0,   0,   0, 0}
    };

    unsigned char img_correct[5][5] = {
        {  0,   0,   0,   0,  0},
        {  0, 153, 204, 153, 51},
        { 51, 204, 255, 204, 51},
        { 51, 153, 204, 153,  0},
        { 0,   0,   0,   0,   0}
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img_filt = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    
    filter_3(img, img_filt, blur_filt, &cpar);
    fail_unless(images_equal(img_filt, img_correct, 5, 5));
    free(img_filt);
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("Image processing");

    TCase *tc = tcase_create ("General filter");
    tcase_add_test(tc, test_general_filter);
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

