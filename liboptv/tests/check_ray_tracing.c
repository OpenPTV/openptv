/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "ray_tracing.h"

START_TEST(test_ray_tracing)
{
    
    fail_unless(1 == 1);    
    
}
END_TEST


START_TEST(test_rotation_matrix)
{
    Exterior Ex_correct, Ex; 
    
    Ex_correct.x0 = 0.0;
    Ex_correct.y0 = 0.0;
    Ex_correct.z0 = 0.0;
    Ex_correct.omega = 0.0;
    Ex_correct.phi = 0.0;
    Ex_correct.kappa = 0.0;
    Ex_correct.dm[0][0] = 1.0;
    Ex_correct.dm[0][1] = 0.0;
    Ex_correct.dm[0][2] = 0.0;
    Ex_correct.dm[1][0] = 0.0;
    Ex_correct.dm[1][1] = 1.0;
    Ex_correct.dm[1][2] = 0.0;
    Ex_correct.dm[2][0] = 0.0;
    Ex_correct.dm[2][1] = 0.0;
    Ex_correct.dm[2][2] = 1.0;
    
	Ex.x0 = 0.0;
    Ex.y0 = 0.0;
    Ex.z0 = 0.0;
    Ex.omega = 0.0;
    Ex.phi = 0.0;
    Ex.kappa = 0.0;
    /*
    Ex.dm[0][0] = 0.0;
    Ex.dm[0][1] = 0.0;
    Ex.dm[0][2] = 0.0;
    Ex.dm[1][0] = 0.0;
    Ex.dm[1][1] = 0.0;
    Ex.dm[1][2] = 0.0;
    Ex.dm[2][0] = 0.0;
    Ex.dm[2][1] = 0.0;
    Ex.dm[2][2] = 0.0;
    */
    
    rotation_matrix(&Ex);
    
    /* 
    printf("%s\n", "Exterior:");
    printf("%lf\n",Ex.x0);
    printf("%lf\n",Ex.y0);
    printf("%lf\n",Ex.z0);
    printf("%lf\n",Ex.omega);
    printf("%lf\n",Ex.phi);
    printf("%lf\n",Ex.kappa);
    printf("%lf\n",Ex.dm[0][0]);
    printf("%lf\n",Ex.dm[0][1]);
    printf("%lf\n",Ex.dm[0][2]);
    printf("%lf\n",Ex.dm[1][0]);
    printf("%lf\n",Ex.dm[1][1]);
    printf("%lf\n",Ex.dm[1][2]);
    printf("%lf\n",Ex.dm[2][0]);
    printf("%lf\n",Ex.dm[2][1]);
    printf("%lf\n",Ex.dm[2][2]);
    */
    
    
    
    fail_unless(compare_exterior(&Ex, &Ex_correct));    
    
}
END_TEST




Suite* fb_suite(void) {
    Suite *s = suite_create ("Ray tracing");

    TCase *tc = tcase_create ("demo test");
    tcase_add_test(tc, test_ray_tracing);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Test rotation matrix");
    tcase_add_test(tc, test_rotation_matrix);
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

