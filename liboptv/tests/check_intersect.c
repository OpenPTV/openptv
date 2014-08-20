/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"
#include "intersect.h"

#define EPS 1E-5


START_TEST(test_intersect_rt)
{
    /* First, test parallel case: */
    double pos1[] = {0.0, 0.0, 0.0};
    double pos2[] = {0.0, 0.0, 0.0};
    double vec1[] = {0.0, 0.0, 1.0};
    double vec2[] = {0.0, 0.0, 1.0};
    
    double X, Y, Z;

    intersect_rt (pos1, vec1, pos2, vec2, &X,&Y,&Z);
				   
    ck_assert_msg( fabs(X - 1e6) < EPS && 
                   fabs(Y - 1e6) < EPS && 
                   fabs(Z - 1e6)  < EPS,
             "Was expecting X,Y,Z to be 1e6 but found %f %f %f\n", X,Y,Z);
     
    /* Test some intersection */         
    vec1[1] = -0.707;
    vec2[1] = 0.707;
    pos1[0] = 1.0;
    pos2[1] = 1.0;
      
    intersect_rt (pos1, vec1, pos2, vec2, &X,&Y,&Z);
				   
    ck_assert_msg( fabs(X - 0.5) < EPS && 
                   fabs(Y - 0.5) < EPS && 
                   fabs(Z + 0.707214)  < EPS,
             "Was expecting X,Y,Z to be 1e6 but found %f %f %f\n", X,Y,Z);
    


}
END_TEST



Suite* fb_suite(void) {
    Suite *s = suite_create ("intersect");
 
    TCase *tc = tcase_create ("intersect test");
    tcase_add_test(tc, test_intersect_rt);
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

