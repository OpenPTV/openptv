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

/*  Check equality of (parts of) images.
    
    Arguments:
    unsigned char *img1, *img2 - images to compare.
    int w, h - width and height of images in pixels
    int offset - start comparing after this many elements.
    int discard - don't compare this many elements from the end.
    
    Returns:
    1 if equal, 0 if not.
*/
int images_equal(unsigned char *img1, unsigned char *img2, 
    int w, int h, int offset, int discard) 
{
    int pix;
    
    for (pix = offset; pix < w*h - discard; pix++)
        if (img1[pix] != img2[pix])
            return 0; 
    return 1;
}

START_TEST(test_general_filter)
{
    filter_t blur_filt = {{0, 0.2, 0}, {0.2, 0.2, 0.2}, {0, 0.2, 0}};
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };

    unsigned char img_correct[25] = {
          0,   0,   0,   0,  0,
          0, 153, 204, 153, 51,
         51, 204, 255, 204, 51,
         51, 153, 204, 153,  0,
         0,   0,   0,   0,   0
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img_filt = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    
    filter_3(img, img_filt, blur_filt, &cpar);
    fail_unless(images_equal(img_filt, img_correct, 5, 5, 6, 6));
    free(img_filt);
}
END_TEST

START_TEST(test_mean_filter)
{
    filter_t mean_filt = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img_filt = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    unsigned char *img_mean = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    
    filter_3(img, img_filt, mean_filt, &cpar);
    lowpass_3(img, img_mean, &cpar);
    fail_unless(images_equal(img_filt, img_mean, 5, 5, 6, 6));
    
    free(img_filt);
    free(img_mean);
}
END_TEST

START_TEST(test_box_blur)
{
    /*  A 3x3 box-blur is equivalent to lowpass_3, so that's the comparison
        we'll make here. Only difference is highpass_3 wraps around rows so
        it has different values at the edges. */
    int elem;
    
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img_filt = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    unsigned char *img_mean = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    
    fast_box_blur(1, img, img_filt, &cpar);
    lowpass_3(img, img_mean, &cpar);
    
    /*  set lowpass edge values to 0 so it equals the no-wrap action of 
        the fast box blur */
    for (elem = 0; elem < 5; elem++) {
        img_mean[5*elem] = 0;
        img_mean[5*elem + 4] = 0;
    }
    
    fail_unless(images_equal(img_filt, img_mean, 5, 5, 6, 6));
    
    free(img_filt);
    free(img_mean);
}
END_TEST

START_TEST(test_split)
{

    int elem;
    
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    }, img1[25], img2[25];
    
    unsigned char img_even[10] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0
    };
    
    unsigned char img_odd[10] = {
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0
    };
    
    unsigned char erased_half[15] = {
         2, 2, 2, 2, 2,
         2, 2, 2, 2, 2,
         2, 2, 2, 2, 2
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    memcpy(img1, img, 25);
    memcpy(img2, img, 25);
    
    /* Note: first line of erased half is only erased from the middle, 
       for historic reasons. */
    split(img1, 1, &cpar);
    fail_unless(images_equal(img1, img_odd, 5, 2, 0, 0));
    fail_unless(images_equal(&(img1[10]), erased_half, 5, 3, 2, 0));
    
    split(img2, 2, &cpar);
    fail_unless(images_equal(img2, img_even, 5, 2, 0, 0));
    fail_unless(images_equal(&(img2[10]), erased_half, 5, 3, 2, 0));
}
END_TEST

START_TEST(test_subtract_img)
{

    int elem;
    
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };
    
    unsigned char img_zero[25] = {
         0,   0,   0,   0, 0,
         0,   0,   0,   0, 0,
         0,   0,   0,   0, 0,
         0,   0,   0,   0, 0,
         0,   0,   0,   0, 0
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img1 = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    // memcpy(img1, img, 25);
    
    subtract_img(img, img_zero, img1, &cpar);
    fail_unless(images_equal(img1, img, 5, 5, 0, 0));
    
    unsigned char *img2 = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    subtract_img(img, img1, img2, &cpar);
    fail_unless(images_equal(img2, img_zero, 5, 5, 0, 0));
    
    free(img1);
    free(img2);
}
END_TEST

START_TEST(test_subtract_mask)
{

    int elem;
    
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };
    
    unsigned char img_mask1[25] = {
         1,   1,   1,   1,  1,
         1,   1,   1,   1,  1,
         1,   1,   1,   1,  1,
         1,   1,   1,   1,  1,
         1,   1,   1,   1,  1
    };
    
    unsigned char img_mask2[25] = {
         1,   1,   1,   1,  1,
         1,   1,   1,   1,  1,
         1,   1,   0,   1,  1,
         1,   1,   1,   1,  1,
         1,   1,   1,   1,  1
    };

    unsigned char img_correct[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255,  0,  255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };


    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img_new = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    
    subtract_mask(img, img_mask1, img_new, &cpar);
    fail_unless(images_equal(img_new, img, 5, 5, 0, 0));
    
    subtract_mask(img, img_mask2, img_new, &cpar);
    fail_unless(images_equal(img_new, img_correct, 5, 5, 0, 0));
    
    free(img_new);
}
END_TEST

START_TEST(test_copy_img)
{

    int elem;
    
    unsigned char img[30] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0,
         1,   1,   1,   1, 1
    };

    control_par cpar = {
        .imx = 6,
        .imy = 5,
    };
    
    unsigned char *img_new = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    unsigned char *img1 = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
            
    copy_images(img, img_new, &cpar);
    fail_unless(images_equal(img_new, img, cpar.imx, cpar.imy, 0, 0));
    
    memcpy(img1,img,cpar.imx*cpar.imy);
    fail_unless(images_equal(img_new, img1, cpar.imx, cpar.imy, 0, 0));
        
    free(img_new);
}
END_TEST

START_TEST(test_highpass)
{
    unsigned char img[25] = {
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    };
    
    unsigned char img_correct[25] = {
         0,   0,   0,   0, 0,
         0, 142, 85, 142, 0,
         0, 85, 0, 85, 0,
         0, 142, 85, 142, 0,
         0,   0,   0,   0, 0
    };

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };
    
    unsigned char *img_hp = (unsigned char *) malloc(cpar.imx*cpar.imy* \
        sizeof(unsigned char));
    
    prepare_image(img, img_hp, 1, 0, 0, &cpar); 
       
    fail_unless(images_equal(img_hp, img_correct, 5, 5, 6, 6));

    free(img_hp);
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("Image processing");

    TCase *tc = tcase_create ("General filter");
    tcase_add_test(tc, test_general_filter);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Mean (lowpass) filter");
    tcase_add_test(tc, test_mean_filter);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Fast box blur");
    tcase_add_test(tc, test_box_blur);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Split image");
    tcase_add_test(tc, test_split);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Subtract image");
    tcase_add_test(tc, test_subtract_img);
    suite_add_tcase (s, tc);
 
    tc = tcase_create ("Subtract mask");
    tcase_add_test(tc, test_subtract_mask);
    suite_add_tcase (s, tc);  
    
    tc = tcase_create ("Copy images");
    tcase_add_test(tc, test_copy_img);
    suite_add_tcase (s, tc);  

    tc = tcase_create ("High-pass");
    tcase_add_test(tc, test_highpass);
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

