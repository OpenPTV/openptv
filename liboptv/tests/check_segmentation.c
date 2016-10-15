/* Unit tests for reading and writing parameter files. */

#include <check.h>
#include "segmentation.h"
#include "tracking_frame_buf.h"


START_TEST(test_peak_fit)
    {
    int i,j, ntargets;
    unsigned char tmpimg[5][5] = {
        { 0,   0,   0,   0, 0},
        { 0, 255, 255, 255, 0},
        { 0, 255, 255, 255, 0},
        { 0, 255, 255, 255, 0},
        { 0,   0,   0,   0, 0}
    };


    target pix[1024];

    control_par cpar = {
        .imx = 5,
        .imy = 5,
    };

    unsigned char* img;
    img = calloc(cpar.imx*cpar.imy, sizeof(unsigned char));
    for (i=0;i<cpar.imy;i++) for(j=0;j<cpar.imx;j++) {img[i*cpar.imx+j]=tmpimg[i][j];}



    target_par targ_par= {
        .gvthres = {250, 100, 20, 20}, 
        .discont = 5,
        .nnmin = 1, .nnmax = 10,
        .nxmin = 1, .nxmax = 10,
        .nymin = 1, .nymax = 10, 
        .sumg_min = 12, 
        .cr_sz = 13 };

                
    ntargets = peak_fit(img, &targ_par, 0, cpar.imx, 0, cpar.imy, &cpar, 0, pix);
    fail_unless(ntargets == 1);
    fail_unless(pix[0].n == 9);

    /* test the two objects */
     unsigned char tmpimg1[5][5] = {
        { 0,   0,   0,   0, 0},
        { 0, 255, 0, 0, 0},
        { 0, 0, 0, 0, 0},
        { 0, 0, 0, 251, 0},
        { 0,   0,   0,   0, 0}
    };

    unsigned char* img1;
    img1 = calloc(cpar.imx*cpar.imy, sizeof(unsigned char));
    for (i=0;i<cpar.imy;i++) for(j=0;j<cpar.imx;j++) {img1[i*cpar.imx+j]=tmpimg1[i][j];}

    ntargets = peak_fit(img1, &targ_par, 0, cpar.imx, 0, cpar.imy, &cpar, 1,
        pix);
    fail_unless(ntargets == 2);

    targ_par.gvthres[1] = 252;
    ntargets = peak_fit((unsigned char *)img1, &targ_par, 0, cpar.imx,
        0, cpar.imy, &cpar, 1, pix);
    fail_unless(ntargets == 1);

    }
END_TEST

START_TEST(test_targ_rec)
{
    int i,j,ntargets;
    unsigned char tmpimg[5][5] = {
        { 0,   0,   0,   0, 0},
        { 0, 255, 255, 255, 0},
        { 0, 255, 255, 255, 0},
        { 0, 255, 255, 255, 0},
        { 0,   0,   0,   0, 0}
    };
    
    target pix[1024];
    
    control_par cpar = {
        .imx = 5,
        .imy = 5,
    }; 


    target_par targ_par= { 
        .gvthres = {250, 100, 20, 20}, 
        .discont = 5,
        .nnmin = 1, .nnmax = 10,
        .nxmin = 1, .nxmax = 10,
        .nymin = 1, .nymax = 10, 
        .sumg_min = 12, 
        .cr_sz = 13 };
    
    unsigned char* img;
    img = calloc(cpar.imx*cpar.imy, sizeof(unsigned char));
    for (i=0;i<cpar.imy;i++) for(j=0;j<cpar.imx;j++) {img[i*cpar.imx+j]=tmpimg[i][j];}
    
    ntargets = targ_rec (img, &targ_par, 0, cpar.imx, 0, cpar.imy, &cpar, 0, pix);
    fail_unless(ntargets == 1);
    fail_unless(pix[0].n == 9);
    fail_unless(pix[0].tnr == CORRES_NONE);
   
    /* test the two objects */
    unsigned char tmpimg1[5][5] = {
        { 0,   0,   0,   0, 0},
        { 0, 255, 0, 0, 0},
        { 0, 0, 0, 0, 0},
        { 0, 0, 0, 251, 0},
        { 0,   0,   0,   0, 0}
    };
    
    unsigned char* img1;
    img1 = calloc(cpar.imx*cpar.imy, sizeof(unsigned char));
    for (i=0;i<cpar.imy;i++) for(j=0;j<cpar.imx;j++) {img1[i*cpar.imx+j]=tmpimg1[i][j];}
    
    ntargets = targ_rec (img1, &targ_par, 0, cpar.imx, 0, cpar.imy, &cpar, 1,
        pix);
    fail_unless(ntargets == 2);
   
    targ_par.gvthres[1] = 252; 
    ntargets = targ_rec ((unsigned char *)img1, &targ_par, 0, cpar.imx, 
        0, cpar.imy, &cpar, 1, pix);
    fail_unless(ntargets == 1);

    /* Trip a segfault writing over the edge. */
    img1[4*cpar.imx+4] = 255;
    ntargets = targ_rec (img1, &targ_par, 0, cpar.imx, 0, cpar.imy, &cpar, 1,
        pix);
    /* If execution reached here, test passed. */
}
END_TEST



Suite* fb_suite(void) {
    Suite *s = suite_create ("Segmentation");
    
    TCase *tc = tcase_create ("Target recording");
    tcase_add_test(tc, test_targ_rec);
    suite_add_tcase (s, tc);

    tc = tcase_create ("Peak Fitting");
    tcase_add_test(tc, test_peak_fit);
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

