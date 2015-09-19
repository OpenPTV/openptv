/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"
#include "ray_tracing.h"
#include "multimed.h"
#include "imgcoord.h"

#define EPS 1E-4



START_TEST(test_img_coord)
{

  fail_if(1);    
    
}
END_TEST


START_TEST(test_img_xy)
{
  
      
    
}
END_TEST


START_TEST(test_img_xy_mm_geo)
{


      
    
}
END_TEST



START_TEST(test_imgcoord)
{


    
}
END_TEST
            






Suite* fb_suite(void) {
    Suite *s = suite_create ("imgcoord");
 
    TCase *tc = tcase_create ("imgcoord_test");

    tcase_add_test(tc, test_img_coord);
    tcase_add_test(tc, test_img_xy);    
    tcase_add_test(tc, test_img_xy_mm_geo);
    tcase_add_test(tc, test_imgcoord);     
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
