/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "ray_tracing.h"


#define EPS 1E-5


START_TEST(test_norm_cross)
{
    
    double n[3];
    
    // test simple cross-product normalized to unity
    
    double a[] = {1.0, 0.0, 0.0};
    double b[] = {0.0, 2.0, 0.0};
    
    norm_cross(a,b,&n[0],&n[1],&n[2]);
    fail_unless( (n[0] == 0.0) && (n[1] == 0.0) && (n[2] == 1.0));
    
    
    // test negative values in the output
    
    norm_cross(b,a,&n[0],&n[1],&n[2]);
    // fail_unless( (n[0] == 0.0) && (n[1] == 0.0) && (n[2] == -1.0));
    
    
    ck_assert_msg( fabs(n[0] - 0.0) < EPS &&
                  fabs(n[1] - 0.0) < EPS &&
                  fabs(n[2] - -1.0)  < EPS,
                  "Was expecting n to be 0., 0., -1. but found %f %f %f\n", n[0],n[1],n[2]);
    
    
    // our norm_cross had a bug when multiplying the parallel vectors
    // it was returning nan instead of 0.0
    // fixed Aug. 3, 2013, see in ray_tracing.c
    
    norm_cross(a,a,&n[0],&n[1],&n[2]);
    // fail_unless( (n[0] == 0.0) && (n[1] == 0.0) && (n[2] == 0.0));
    
    ck_assert_msg( fabs(n[0] - 0.0) < EPS &&
                  fabs(n[1] - 0.0) < EPS &&
                  fabs(n[2] - 0.0)  < EPS,
                  "Was expecting n to be 0., 0., 0. but found %f %f %f\n", n[0],n[1],n[2]);
    
    
}
END_TEST


START_TEST(test_dot)
{
    
    double d;
    
    // test simple cross-product normalized to unity
    
    double a[] = {1.0, 0.0, 0.0};
    double b[] = {0.0, 2.0, 0.0};
    
    dot(a,b,&d);
    
    //fail_unless( d == 0.0 );
    ck_assert_msg( fabs(d - 0.0) < EPS,
                  "Was expecting d to be 0.0 but found %f \n", d);
    
    b[0] = 2.0;
    b[1] = 2.0;
    b[2] = 0.0;
    
    dot(b,a,&d);
    // fail_unless( d == 2.0 );
    ck_assert_msg( fabs(d - 2.0) < EPS,
                  "Was expecting d to be 2.0 but found %f \n", d);
    
}
END_TEST


START_TEST(test_modu)
{
    double a[]= {10.0, 0.0, 0.0};
    double m;
    
    modu(a,&m);
    
    // fail_unless( m == 10.0);
    ck_assert_msg( fabs(m - 10.0) < EPS,
                  "Was expecting m to be 10.0 but found %f \n", m);
    
}
END_TEST




START_TEST(test_ray_tracing)
{
	double     x1, y1, a,b,c, X1,Y1,Z1;
	
	x1 = 100.;
	y1 = 100.;
	
    
    
	Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0,
        {{1.0, 0.2, -0.3},
            {0.2, 1.0, 0.0},
            {-0.3, 0.0, 1.0}}};
    
    Interior test_I = {0., 0., 100.};
    Glass test_G = {0.0001, 0.00001, 1.};
    // ap_52 test_addp = {0., 0., 0., 0., 0., 1., 0.};
    
    /*
     typedef struct {
     int  	nlay;
     double  n1;
     double  n2[3];
     double  d[3];
     double  n3;
     int     lut;
     } mm_np;
     */
    
    mm_np test_mmp = {
        3,
        1.,
        {1.49,0.,0.},
        {5.,0.,0.},
        1.33,
        1.};
    
    ray_tracing_v2 (x1,y1, test_Ex, test_I, test_G, test_mmp, \
                    &X1, &Y1, &Z1, &a, &b, &c);
    
    ck_assert_msg( fabs(X1 - 110.406944) < EPS &&
                  fabs(Y1 - 88.325788) < EPS &&
                  fabs(Z1 - 0.988076)  < EPS,
                  "Was expecting X1,Y1,Z1 to be 110.407 88.326 0.988 but found %f %f %f\n", \
                  X1,Y1,Z1);
    ck_assert_msg( fabs(a - 0.387960) < EPS && 
                  fabs(b - 0.310405) < EPS && 
                  fabs(c - -0.867834)  < EPS,
                  "Was expecting a,b,c to be 0.387960 0.310405 -0.867834 but found %f %f %f\n", \
                  a,b,c);          
    
    
    
}
END_TEST




Suite* fb_suite(void) {
    Suite *s = suite_create ("Ray tracing");
    
    TCase *tc = tcase_create ("ray tracing test");
    tcase_add_test(tc, test_norm_cross);
    tcase_add_test(tc, test_dot);
    tcase_add_test(tc, test_modu);
    tcase_add_test(tc, test_ray_tracing);
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

