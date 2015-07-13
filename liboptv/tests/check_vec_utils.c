/*  Unit tests for the vector utilities. Uses the Check
    framework: http://check.sourceforge.net/
    
    To run it, type "make check" when in the top C directory, src_c/
    If that doesn't run the tests, use the Check tutorial:
    http://check.sourceforge.net/doc/check_html/check_3.html
*/

#include <check.h>
#include <stdlib.h>
#include <math.h>
#include "vec_utils.h"

#define EPS 1E-5


START_TEST(test_dot)
{
    // test simple dot-product
    double d;
    
    vec3d a = {1.0, 0.0, 0.0};
    vec3d b = {0.0, 2.0, 0.0};
    
    d = vec_dot(a, b);
    ck_assert_msg( fabs(d - 0.0) < EPS,
             "Was expecting d to be 0.0 but found %f \n", d);

    vec_set(b, 2.0, 2.0, 0.0);
    d = vec_dot(b, a);
    ck_assert_msg( fabs(d - 2.0) < EPS,
             "Was expecting d to be 2.0 but found %f \n", d);

}
END_TEST

START_TEST(test_vec_init)
{
    int i;
    vec3d p;
    vec_init(p);

    for (i = 0; i < 3; i++)
        fail_unless(is_empty(p[i]));
}
END_TEST

START_TEST(test_vec_cmp)
{
    /* Putting it first because it's useful for the other tests */
    vec3d v1 = {1., 2., 3.}, v2 = {4., 5., 6.};
    
    fail_unless(vec_cmp(v1, v1));
    fail_if(vec_cmp(v1, v2));
}
END_TEST

START_TEST(test_vec_copy)
{
    vec3d src = {1., 2., 3.}, dst;
    vec_copy(dst, src);

    fail_unless(vec_cmp(dst, src));
}
END_TEST

START_TEST(test_vec_subt)
{
    vec3d sub = {1., 2., 3.}, from = {4., 5., 6.}, res = {3., 3., 3.}, out;
    vec_subt(from, sub, out);

    fail_unless(vec_cmp(out, res));
}
END_TEST

START_TEST(test_diff_norm)
{
    int i;
    vec3d vec1 = {1., 2., 3.}, vec2 = {4., 5., 6.};
    fail_unless(vec_diff_norm(vec1, vec2) == sqrt(3)*3);
}
END_TEST

START_TEST(test_vec_set)
{
    vec3d res = {1., 2., 3.}, dest;
    vec_set(dest, 1., 2., 3.);
    fail_unless(vec_cmp(dest, res));
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("lsqadj");
 
    TCase *tc = tcase_create ("Vector dot product");
    tcase_add_test(tc, test_dot);
    suite_add_tcase (s, tc);   
    
    tc = tcase_create("Init vec3d");
    tcase_add_test(tc, test_vec_init);
    suite_add_tcase (s, tc);
    
    tc = tcase_create("Copy vec3d");
    tcase_add_test(tc, test_vec_copy);
    suite_add_tcase (s, tc);

    tc = tcase_create("Subtract vec3d");
    tcase_add_test(tc, test_vec_subt);
    suite_add_tcase (s, tc);

    tc = tcase_create("Compare vec3d");
    tcase_add_test(tc, test_vec_cmp);
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

