/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "ray_tracing.h"


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

fail_unless( (n[0] == 0.0) && (n[1] == 0.0) && (n[2] == -1.0));

}
END_TEST


START_TEST(test_dot)
{

double d;

// test simple cross-product normalized to unity

double a[] = {1.0, 0.0, 0.0};
double b[] = {0.0, 2.0, 0.0};

dot(a,b,&d);

fail_unless( d == 0.0 );

b[0] = 2.0;
b[1] = 2.0;
b[2] = 0.0;

dot(b,a,&d);
fail_unless( d == 2.0 );

}
END_TEST


START_TEST(test_modu)
{
double a[]= {10.0, 0.0, 0.0};
double m;

modu(a,&m);

fail_unless( m == 10.0);

}
END_TEST



START_TEST(test_ray_tracing)
{
    
    fail_unless(1 == 1);    
    
}
END_TEST







Suite* fb_suite(void) {
    Suite *s = suite_create ("Ray tracing");
 
    TCase *tc = tcase_create ("norm_cross test");
    tcase_add_test(tc, test_norm_cross);
    suite_add_tcase (s, tc);   
    
    tc = tcase_create ("test dot");
    tcase_add_test(tc, test_dot);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("test modu");
    tcase_add_test(tc, test_modu);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("demo test");
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

