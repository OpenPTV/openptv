/* Unit tests for finding image coordinates of 3D position. */

#include <check.h>
#include <stdlib.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "imgcoord.h"
#include "vec_utils.h"

#define EPS 1E-6

START_TEST(test_flat_centered_cam)
{
    /*  When the image plane is centered on the axis. and the camera looks to
        a straight angle (e.g. along an axis), the image position can be 
        gleaned from simple geometry.
    */
    vec3d pos = {10, 5, -20};
    Calibration cal = {
        .ext_par = {0, 0, 40, 0, 0, 0, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        .int_par = {0, 0, 10},
        .glass_par = {0, 0, 20},
        .added_par = {0, 0, 0, 0, 0, 1, 0}
    };
    mm_np mm = { /* All in air, simplest case. */
        .nlay = 1,
        .n1 = 1,
        .n2 = {1,0,0},
        .n3 = 1,
        .d = {1,0,0}
    };
    double x, y; /* Output variables */
    
    flat_image_coord(pos, &cal, &mm, &x, &y);
    fail_unless(x == 10./6.);
    fail_unless(x == 2*y);
}
END_TEST

START_TEST(test_shifted_sensor)
{
    /*  When the image plane is centered on the axis. and the camera looks to
        a straight angle (e.g. along an axis), the image position can be 
        gleaned from simple geometry.
    */
    vec3d pos = {10, 5, -20};
    Calibration cal = {
        .ext_par = {0, 0, 40, 0, 0, 0, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        .int_par = {0.1, 0.1, 10},
        .glass_par = {0, 0, 20},
        .added_par = {0, 0, 0, 0, 0, 1, 0}
    };
    mm_np mm = { /* All in air, simplest case. */
        .nlay = 1,
        .n1 = 1,
        .n2 = {1,0,0},
        .n3 = 1,
        .d = {1,0,0}
    };
    double x, y; /* Output variables */
    
    img_coord(pos, &cal, &mm, &x, &y);
    fail_unless(x == 10./6. + 0.1);
    fail_unless(x == 2*(y - 0.1) + 0.1);
}
END_TEST

START_TEST(test_flat_decentered_cam)
{
    /*  When the camera axis goes through the origin, the image point is (0, 0)
        for centered internal parameters.
    */
    vec3d pos = {10, 0, -20};
    Calibration cal = {
        .ext_par = {-20, 0, 40, 0, -atan(0.5), 0, 
            {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        .int_par = {0, 0, 10},
        .glass_par = {0, 0, 20},
        .added_par = {0, 0, 0, 0, 0, 1, 0}
    };
    mm_np mm = { /* All in air, simplest case. */
        .nlay = 1,
        .n1 = 1,
        .n2 = {1,0,0},
        .n3 = 1,
        .d = {1,0,0}
    };
    double x, y; /* Output variables */
    
    rotation_matrix(&cal.ext_par);
    flat_image_coord(pos, &cal, &mm, &x, &y);
    fail_unless(fabs(x) < EPS);
    fail_unless(fabs(y) < EPS);
}
END_TEST

START_TEST(test_flat_multilayer)
{
    /*  When the camera axis goes through the origin, the image point is (0, 0)
        for centered internal parameters. That's true even for varying 
        refractive indices, if the glass normal is still parallel to the camera
        to point line.
    */
    double angle = atan(0.5);
    vec3d pos = {10, 0, -20};
    Calibration cal = {
        .ext_par = {-20, 0, 40, 0, -angle, 0, 
            {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        .int_par = {0, 0, 10},
        .glass_par = {-20*sin(angle), 0, 20*cos(angle)},
        .added_par = {0, 0, 0, 0, 0, 1, 0}
    };
    mm_np mm = { /* All in air, except the glass. */
        .nlay = 1,
        .n1 = 1,
        .n2 = {1.5,0,0},
        .n3 = 1,
        .d = {1,0,0}
    };
    double x, y; /* Output variables */
    
    rotation_matrix(&cal.ext_par);
    flat_image_coord(pos, &cal, &mm, &x, &y);
    fail_unless(fabs(x) < EPS);
    fail_unless(fabs(y) < EPS);
}
END_TEST

START_TEST(test_distorted_centered_cam)
{
    /*  When the image plane is centered on the axis. and the camera looks to
        a straight angle (e.g. along an axis), the image position can be 
        gleaned from simple geometry. Distortion can be predicted.
    */
    vec3d pos = {10, 5, -20};
    Calibration cal = {
        .ext_par = {0, 0, 40, 0, 0, 0, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        .int_par = {0, 0, 10},
        .glass_par = {0, 0, 20},
        .added_par = {-0.01, 0, 0, 0, 0, 1, 0} /* barrel distortion */
    };
    mm_np mm = { /* All in air, simplest case. */
        .nlay = 1,
        .n1 = 1,
        .n2 = {1,0,0},
        .n3 = 1,
        .d = {1,0,0}
    };
    double x, y; /* Output variables */
    double r; /* radial distortion prediction */
    
    img_coord(pos, &cal, &mm, &x, &y);
    r = vec_norm3d(10./6., 5./6., 0);
    fail_unless(x == 10./6.*(1 - 0.01*r*r));
    fail_unless(x == 2*y);
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("Imgcoord");
 
    TCase *tc = tcase_create ("Image coordinates all-air");
    tcase_add_test(tc, test_flat_centered_cam);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Image coordinates all-air camera moved");
    tcase_add_test(tc, test_flat_decentered_cam);     
    suite_add_tcase (s, tc); 
      
    tc = tcase_create ("Multilayer image coordinates, no distortion");
    tcase_add_test(tc, test_flat_multilayer);     
    suite_add_tcase (s, tc); 
    
    tc = tcase_create ("Distorted image coordinates");
    tcase_add_test(tc, test_distorted_centered_cam);
    suite_add_tcase (s, tc);
    
    tc = tcase_create ("Shifted sensor not ignored");
    tcase_add_test(tc, test_shifted_sensor);
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

