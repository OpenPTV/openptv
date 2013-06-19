/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include "parameters.h"

START_TEST(test_ray_tracing)
{
    
    fail_unless(1 == 1);    
    
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("Ray tracing");

    TCase *tc = tcase_create ("demo test");
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

