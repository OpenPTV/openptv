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

void old_metric_to_pixel();
void old_pixel_to_metric();


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



START_TEST(test_distort_brown_affin)
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


START_TEST(test_correct_brown_affin)
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






Suite* fb_suite(void) {
    Suite *s = suite_create ("trafo");
    TCase *tc = tcase_create ("trafo_test");
    tcase_add_test(tc, test_old_metric_to_pixel);
    tcase_add_test(tc, test_metric_to_pixel);
    tcase_add_test(tc, test_pixel_to_metric );
    tcase_add_test(tc, test_old_pixel_to_metric);
    tcase_add_test(tc, test_distort_brown_affin);
    tcase_add_test(tc, test_correct_brown_affin);
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

