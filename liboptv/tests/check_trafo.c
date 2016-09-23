/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"
#include "ray_tracing.h"
#include "trafo.h"

#define EPS 1E-5


START_TEST(test_old_metric_to_pixel)
{
    /* input */
    double xc = 0.0; // [mm]
    double yc = 0.0; // [mm]
    int imx = 1024; // standard image size
    int imy = 1008; 
    double pix_x = 0.010; // 10 micron pixel size
    double pix_y = 0.010;
    int field = 0; // simple image format, not interlaced
    
    /* output */
    double xp, yp;        
        
    
    old_metric_to_pixel (&xp, &yp, xc, yc, imx, imy, pix_x, pix_y,  field);    
    
    
     ck_assert_msg( fabs(xp - 512.0) < EPS && 
                    fabs(yp - 504.0) < EPS,
         "Expected 512.0, 504.0, but got %f %f\n", xp, yp);
         
    xc = 1.0;
    yc = 0.0;
    
    old_metric_to_pixel (&xp, &yp, xc, yc, imx, imy, pix_x, pix_y,  field);    
    
    
     ck_assert_msg( fabs(xp - 612.0) < EPS && 
                    fabs(yp - 504.0) < EPS,
         "Expected 612.0, 504.0, but got %f %f\n", xp, yp);
         
    xc = 0.0;
    yc = -1.0;
    
    old_metric_to_pixel (&xp, &yp, xc, yc, imx, imy, pix_x, pix_y,  field);    
    
    
     ck_assert_msg( fabs(xp - 512.0) < EPS && 
                    fabs(yp - 604.0) < EPS,
         "Expected 512.0, 604.0, but got %f %f\n", xp, yp);
  
    
}
END_TEST



START_TEST(test_metric_to_pixel)
{
    /* input */
    double xc = 0.0; // [mm]
    double yc = 0.0; // [mm]
    control_par cpar;
       
    /* output */
    double xp, yp;     

    
    cpar.imx = 1024; 
    cpar.imy = 1008;
    cpar.pix_x = 0.01;
    cpar.pix_y = 0.01;
    cpar.chfield = 0;
    
       
    
    metric_to_pixel (&xp, &yp, xc, yc, &cpar);    
    
    
     ck_assert_msg( fabs(xp - 512.0) < EPS && 
                    fabs(yp - 504.0) < EPS,
         "Expected 512.0, 504.0, but got %f %f\n", xp, yp);
         
    xc = 1.0;
    yc = 0.0;
    
    metric_to_pixel (&xp, &yp, xc, yc, &cpar);    
    
    
     ck_assert_msg( fabs(xp - 612.0) < EPS && 
                    fabs(yp - 504.0) < EPS,
         "Expected 612.0, 504.0, but got %f %f\n", xp, yp);
         
    xc = 0.0;
    yc = -1.0;
    
    metric_to_pixel (&xp, &yp, xc, yc, &cpar);     
    
    
     ck_assert_msg( fabs(xp - 512.0) < EPS && 
                    fabs(yp - 604.0) < EPS,
         "Expected 512.0, 604.0, but got %f %f\n", xp, yp);
  
    
}
END_TEST

START_TEST(test_old_pixel_to_metric)
{
    /* input */
    double xc = 0.0; // [mm]
    double yc = 0.0; // [mm]
    int imx = 1024; // standard image size
    int imy = 1008; 
    double pix_x = 0.010; // 10 micron pixel size
    double pix_y = 0.010;
    int field = 0; // simple image format, not interlaced
    
    /* output */
    double xp, yp;  
    
    /* compare the xc, yc to the original */
    double xc1, yc1;      
        
    
    old_metric_to_pixel (&xp, &yp, xc, yc, imx, imy, pix_x, pix_y,  field); 
    old_pixel_to_metric (&xc1, &yc1,xp, yp, imx, imy, pix_x, pix_y,  field);   
    
    
     ck_assert_msg( fabs(xc1 - xc) < EPS && 
                    fabs(yc1 - yc) < EPS,
         "Expected %f, %f but got %f %f\n", xc, yc, xc1, yc1);
         
    xc = 1.0;
    yc = 0.0;
    
    old_metric_to_pixel (&xp, &yp, xc, yc, imx, imy, pix_x, pix_y,  field);    
    old_pixel_to_metric (&xc1, &yc1,xp, yp, imx, imy, pix_x, pix_y,  field);   
    
    
     ck_assert_msg( fabs(xc1 - xc) < EPS && 
                    fabs(yc1 - yc) < EPS,
         "Expected %f, %f, but got %f %f\n", xc,yc,xc1, yc1);
         
    xc = 0.0;
    yc = -1.0;
    
    old_metric_to_pixel (&xp, &yp, xc, yc, imx, imy, pix_x, pix_y,  field);    
    old_pixel_to_metric (&xc1, &yc1,xp, yp, imx, imy, pix_x, pix_y,  field);   
    
    
     ck_assert_msg( fabs(xc1 - xc) < EPS && 
                    fabs(yc1 - yc) < EPS,
         "Expected %f, %f, but got %f %f\n", xc,yc,xc1, yc1);
    
}
END_TEST



START_TEST(test_pixel_to_metric)
{
    /* input */
    double xc = 0.0; // [mm]
    double yc = 0.0; // [mm]
    control_par cpar;
       
    /* output */
    double xp, yp;     

    
    cpar.imx = 1024; 
    cpar.imy = 1008;
    cpar.pix_x = 0.01;
    cpar.pix_y = 0.01;
    cpar.chfield = 0;  
    
    /* compare the xc, yc to the original */
    double xc1, yc1;      
        
    
    metric_to_pixel (&xp, &yp, xc, yc, &cpar); 
    pixel_to_metric (&xc1, &yc1, xp, yp, &cpar);   
    
    
     ck_assert_msg( fabs(xc1 - xc) < EPS && 
                    fabs(yc1 - yc) < EPS,
         "Expected %f, %f but got %f %f\n", xc, yc, xc1, yc1);
         
    xc = 1.0;
    yc = 0.0;
    
    metric_to_pixel (&xp, &yp, xc, yc, &cpar); 
    pixel_to_metric (&xc1, &yc1, xp, yp, &cpar);  
    
    
     ck_assert_msg( fabs(xc1 - xc) < EPS && 
                    fabs(yc1 - yc) < EPS,
         "Expected %f, %f, but got %f %f\n", xc,yc,xc1, yc1);
         
    xc = 0.0;
    yc = -1.0;
    
    metric_to_pixel (&xp, &yp, xc, yc, &cpar); 
    pixel_to_metric (&xc1, &yc1, xp, yp, &cpar);   
    
    
     ck_assert_msg( fabs(xc1 - xc) < EPS && 
                    fabs(yc1 - yc) < EPS,
         "Expected %f, %f, but got %f %f\n", xc,yc,xc1, yc1);
    
}
END_TEST



START_TEST(test_shear)
{
    /* input */
    double x = 1.0; // [mm]
    double y = 1.0; // [mm]
    ap_52 ap = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0}; // affine parameters, see calibration

    /* output */
    double xp, yp;          
    
    
    
    distort_brown_affin (x, y, ap, &xp, &yp);
        
     ck_assert_msg( fabs(xp - 0.158529) < EPS && 
                    fabs(yp - 0.540302) < EPS,
         "Expected 0.158529, 0.540302, but got %f %f\n", xp, yp);
           
}
END_TEST


START_TEST(shear_round_trip)
{
    /* input */
    double x = -1.0; // [mm]
    double y = 10.0; // [mm]
    ap_52 ap = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0}; // affine parameters, see calibration

    /* output */
    double xp, yp, x1, y1;           
    
    
    
    distort_brown_affin (x, y, ap, &xp, &yp);
    correct_brown_affin (xp, yp, ap, &x1, &y1);
        
    ck_assert_msg( fabs(x1 - x) < EPS && 
                   fabs(y1 - y) < EPS,
         "Expected %f, %f, but got %f %f\n", x,y,x1,y1);
           
}
END_TEST

START_TEST(dummy_distortion_round_trip)
{
    /* This is the most basic distortion test: if there is no distortion, 
       a point distorted/corrected would come back as the same point, up to
       floating point errors. 
    */
    double x=1., y=1.;
    double xres, yres;
    ap_52 ap = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}; /* no distortion */
    
    distort_brown_affin (x, y, ap, &xres, &yres);
    correct_brown_affin (xres, yres, ap, &xres, &yres);
    
    ck_assert_msg( fabs(xres - x) < EPS && 
                   fabs(yres - y) < EPS,
         "Expected %f, %f, but got %f %f\n", x, y, xres, yres);
}
END_TEST

START_TEST(radial_distortion_round_trip)
{
    /* Less basic distortion test: with radial distortion, a point 
       distorted/corrected would come back as the same point, up to floating 
       point errors and an error from the short iteration. 
    */
    double x=1., y=1.;
    double xres, yres;
    double iter_eps = 1e-2; /* Verified manually with calculator */
    
    /* huge radial distortion */
    ap_52 ap = {0.05, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    distort_brown_affin (x, y, ap, &xres, &yres);
    correct_brown_affin (xres, yres, ap, &xres, &yres);
    
    ck_assert_msg( fabs(xres - x) < iter_eps && 
                   fabs(yres - y) < iter_eps,
         "Expected %f, %f, but got %f %f\n", x, y, xres, yres);
}
END_TEST

START_TEST(dist_flat_round_trip)
{
    /*  Cheks that the order of operations in converting metric flat image to
        distorted image coordinates and vice-versa is correct. 
        
        Distortion value in this one is kept to a level suitable (marginally)
        to an Rmax = 10 mm sensor. The allowed round-trip error is adjusted
        accordingly. Note that the higher the distortion parameter, the worse
        will the round-trip error be, at least unless we introduce more iteration
    */
    double x=10., y=10.;
    double xres, yres;
    double iter_eps = 1e-5; 
    
    Calibration cal = {
        .int_par = {1.5, 1.5, 60.},
        .added_par = {0.0005, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
    };
    
    flat_to_dist(x, y, &cal, &xres, &yres);
    dist_to_flat(xres, yres, &cal, &xres, &yres, 0.00001);
    
    ck_assert_msg( fabs(xres - x) < iter_eps &&
                   fabs(yres - y) < iter_eps,
         "Expected %f, %f, but got %f %f\n", x, y, xres, yres);
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("trafo");
    TCase *tc = tcase_create ("trafo_test");
    tcase_add_test(tc, test_old_metric_to_pixel);
    tcase_add_test(tc, test_metric_to_pixel);
    tcase_add_test(tc, test_pixel_to_metric );
    tcase_add_test(tc, test_old_pixel_to_metric);
    tcase_add_test(tc, test_shear);
    tcase_add_test(tc, shear_round_trip);
    tcase_add_test(tc, dummy_distortion_round_trip);
    tcase_add_test(tc, radial_distortion_round_trip);
    tcase_add_test(tc, dist_flat_round_trip);
    suite_add_tcase (s, tc);   
    return s;
}

int main(void) {
    int number_failed;
    Suite *s = fb_suite ();
    SRunner *sr = srunner_create (s);
    //srunner_run_all (sr, CK_ENV);
    //srunner_run_all (sr, CK_SUBUNIT);
    srunner_run_all (sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

