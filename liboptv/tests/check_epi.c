/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "epi.h"
#include "multimed.h"

#define EPS 1E-5

START_TEST(test_epi_mm_2D)
{
        double x, y;
        vec3d pos, v, out;
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};
    
    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 50.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	1, 
    	1.0, 
    	{1.49, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.33
    };
	
    volume_par test_vpar = {
        {-250., 250.}, {-100., -100.}, {100., 100.}, 0.01, 0.3, 0.3, 0.01, 1.0, 33
        };
        
    /* non-trivial case */
     x = 1.0; 
     y = 10.0;
    
    epi_mm_2D (x, y, &test_cal, &test_mm, &test_vpar, out);

    
    ck_assert_msg(  fabs(out[0] - 0.85858163) < EPS && 
                    fabs(out[1] - 8.58581626) < EPS && 
                    fabs(out[2] - 0.0)  < EPS,
         "\n Expected 0.8586 8.5858 0.0000 \n  \
         but found %10.8f %10.8f %10.8f \n", out[0], out[1], out[2]);
    
    epi_mm_2D (0.0, 0.0, &test_cal, &test_mm, &test_vpar, out);

    
    ck_assert_msg(  fabs(out[0] - 0.0) < EPS && 
                    fabs(out[1] - 0.0) < EPS && 
                    fabs(out[2] - 0.0)  < EPS,
         "\n Expected all NULL \n  \
         but found %10.8f %10.8f %10.8f \n", out[0], out[1], out[2]);
    
}
END_TEST

START_TEST(test_epi_mm)
{

    double x, y, z, xmin, xmax, ymin, ymax;
    vec3d pos, v;

    /* first camera */
    
    Exterior test_Ex_1 = {
        10.0, 0.0, 100.0,
        0.0, -0.01, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};

    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 50.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal_1 = {test_Ex_1, test_I, test_G, test_addp};

    /* second camera at small angle around y axis */
    
    Exterior test_Ex_2 = {
        -10.0, 0.0, 100.0,
        0.0, 0.01, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};

    Calibration test_cal_2 = {test_Ex_2, test_I, test_G, test_addp};

    
    volume_par test_vpar = {
        {-250., 250.}, {-50., -50.}, {50., 50.}, 0.01, 0.3, 0.3, 0.01, 1.0, 33
        };
    
    /* non-trivial case */
     x = 10.0; 
     y = 10.0;
 
     /* void  epi_mm (double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np mmp, volume_par *vpar,
    int i_cam, mmlut *mmLUT,
    double *xmin, double *ymin, double *xmax, double *ymax); */
         
    mm_np test_mm = {
        1, 
        1.0, 
        {1.49, 0.0, 0.0}, 
        {5.0, 0.0, 0.0},
        1.33
    };

        
    epi_mm (x, y, &test_cal_1, &test_cal_2, &test_mm, &test_vpar, \
    &xmin, &xmax, &ymin, &ymax);


    ck_assert_msg(  fabs(xmin -  26.44927852) < EPS && 
                    fabs(xmax - 10.08218486 ) < EPS && 
                    fabs(ymin - 51.60078764) < EPS && 
                    fabs(ymax - 10.04378909)  < EPS,
         "\n Expected 26.44927852 10.08218486 51.60078764 10.04378909 \n  \
         but found %10.8f %10.8f %10.8f %10.8f \n", xmin, xmax, ymin, ymax);

    
}
END_TEST

START_TEST(test_epi_mm_perpendicular)
{

    double x, y, z, xmin, xmax, ymin, ymax;

    /* first camera */
    
    Exterior test_Ex_1 = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};

    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 50.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal_1 = {test_Ex_1, test_I, test_G, test_addp};

    /* second camera at small angle around y axis */
    
    Exterior test_Ex_2 = {
        100.0, 0.0, 0.0,
        0.0, 1.57, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};

    Calibration test_cal_2 = {test_Ex_2, test_I, test_G, test_addp};

    /* all in air */
    mm_np test_mm = {
        1, 
        1.0, 
        {1.0, 0.0, 0.0}, 
        {1.0, 0.0, 0.0},
        1.0,
    };
    
    volume_par test_vpar = {
        {-100., 100.}, {-100., -100.}, {100.0, 100.0}, 0.01, 0.3, 0.3, 0.01, 1.0, 33};
    

    epi_mm (0.0, 0.0, &test_cal_1, &test_cal_2, &test_mm, &test_vpar, \
    &xmin, &xmax, &ymin, &ymax);


    ck_assert_msg(  fabs(xmin + 100.0000) < EPS && 
                    fabs(xmax - 0.0000) < EPS && 
                    fabs(ymin - 100.0000) < EPS && 
                    fabs(ymax - 0.0000)  < EPS,
         "\n Expected -100.0000 0.0000 100.0000 -0.0000 \n  \
         but found %10.8f %10.8f %10.8f %10.8f \n", xmin, xmax, ymin, ymax);

  

}
END_TEST

START_TEST(test_find_candidate)
{

    int i;
    /* set of particles to choose from */

    target test_pix[] = {
        {0, 0.0, -0.2, 5, 1, 2, 10, -999},
        {6, 0.2, 0.0, 10, 8, 1, 20, -999},
        {3, 0.2, 0.8, 10, 3, 3, 30, -999},
        {4, 0.4, -1.1, 10, 3, 3, 40, -999},
        {1, 0.7, -0.1, 10, 3, 3, 50, -999},
        {7, 1.2, 0.3, 10, 3, 3, 60, -999},
        {5, 10.4, 0.1, 10, 3, 3, 70, -999}
    };
    			   
    int num_pix = 7; /* length of the test_pix */

    /* coord_2d is int pnr, double x,y */
    coord_2d test_crd[] = {
        {0, 0.0, 0.0},
        {6, 0.1, 0.1}, /* best candidate, right on the diagonal */
        {3, 0.2, 0.8},
        {4, 0.4, -1.1},
        {1, 0.7, -0.1},
        {7, 1.2, 0.3},
        {5, 10.4, 0.1}
    };

    /* parameters of the particle for which we look for the candidates */
    int n = 10; 
    int nx = 3; 
    int ny = 3;
    int sumg = 100;
    
    candidate test_cand[7];
    
    int count; 
    
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}}};
    
    Interior test_I = {0.0, 0.0, 100.0};
    Glass test_G = {0.0, 0.0, 50.0};
    ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration test_cal = {test_Ex, test_I, test_G, test_addp};
    
    mm_np test_mm = {
    	1, 
    	1.0, 
    	{1.49, 0.0, 0.0}, 
    	{5.0, 0.0, 0.0},
    	1.33,
    };
    
    volume_par test_vpar = {
        {-250., 250.}, {-100., -100.}, {100., 100.}, 0.01, 0.3, 0.3, 0.01, 1.0, 33
    };    
    
    /* prepare test control parameters, basically for pix_x  */
    int cam;
    char img_format[] = "cam%d";
    char cal_format[] = "cal/cam%d.tif";
    control_par *test_cpar, *cpar;
    
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
    
    /* the result is that the sensor size is 12.8 mm x 10.24 mm */
    
    /* epipolar line  */
    double xa = -10.;
    double ya = -10.;
    double xb = 10.;
    double yb = 10.;
    
    count = find_candidate (test_crd, test_pix, num_pix, xa, ya, xb, yb, 
        n, nx, ny, sumg, test_cand, &test_vpar, test_cpar, &test_cal);
    
    free_control_par(test_cpar);
    
    /* expected results: */
    fail_unless(test_cand[0].pnr = 1);
    fail_unless(test_cand[0].tol < EPS);

    /* regression guard */
    double sum_corr = 0;
    for (i = 0; i < count; i++) {
    	sum_corr += test_cand[i].corr;
    }	
   
    ck_assert_msg( fabs(sum_corr - 2625.) < EPS && 
                   (count == 4)  && 
                    fabs(test_cand[3].tol  - 0.565685) < EPS,
         "\n Expected ...  \n  \
         but found %f %d %9.6f \n", sum_corr, count, test_cand[3].tol);	
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("epi");
    
    TCase *tc = tcase_create ("epi_test");
    tcase_add_test(tc, test_epi_mm_2D);
    tcase_add_test(tc, test_epi_mm_perpendicular);
    tcase_add_test(tc, test_epi_mm);    
    tcase_add_test(tc, test_find_candidate);
    
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

