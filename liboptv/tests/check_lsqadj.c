/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"

#define EPS 1E-5

START_TEST(test_matmul)
{

    double a[3] = {1.0,1.0,1.0};
    double b[3] = {0.0,0.0,0.0};
    double d[4][4] = {{1, 2, 3, 99}, {4, 5, 6, 99}, {7, 8, 9, 99},{99, 99, 99, 99}};
    double e[4] = {10, 11, 12, 99};
    double f[3] = {0, 0, 0};
    double expected[3] = {68, 167, 266};
    int i,j,k;
        
        
    Exterior test_Ex = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.2, -0.3}, 
        {0.2, 1.0, 0.0},
        {-0.3, 0.0, 1.0}}};
        
    
    
      matmul((double *)b, (double *) test_Ex.dm, (double *)a, 3,3,1,3,3);
    
    
    
     ck_assert_msg( fabs(b[0] - 0.9) < EPS && 
                    fabs(b[1] - 1.20) < EPS && 
                    fabs(b[2] - 0.700)  < EPS,
         "Was expecting b to be 0.9,1.2,0.7 but found %f %f %f\n", b[0],b[1],b[2]);
    
    


        matmul((double *)f, (double *) d, (double *) e, 3, 3, 1, 4, 4);
                
    for (i=0; i<3; i++){
                 ck_assert_msg(fabs(f[i] - expected[i]) < EPS, "wrong item \
                 [%d] %f instead of %f", i,a[i],expected[i]);
                 }
    
    
    
}
END_TEST



START_TEST(test_ata)
{

    double a[4][3] = {{1, 0, 1}, {2, 2, 4},{1, 2, 3}, {2, 4, 3}};
    double b[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double expected[3][3] = {{10, 14, 18}, {14, 24, 26},{18, 26, 35}};
    
    int i, j;
    
        
    ata((double*) a, (double *) b, 4, 3, 3);
    
    for (i=0; i<3; i++){
            for (j=0; j<3; j++){
                 ck_assert_msg(fabs(b[i][j] - expected[i][j]) < EPS, "wrong item \
                 [%d][%d] %f instead of %f", i,j,b[i][j],expected[i][j]);
                 }
            }
    
}
END_TEST

START_TEST(test_atl)
{

    double a[4][3] = {{1, 0, 1}, {2, 2, 4},{1, 2, 3}, {2, 4, 3}};
    double l[4] = {1,2,3,4};
    double u[3];
    double expected[3] = {16, 26, 30};
    
    int i, j;
    
    for (i=0; i<3; i++){
                 u[i] = 0.0;
            }
            
    atl((double *)u, (double *)a, (double *)l, 4, 3, 3);
    
    
    for (i=0; i<3; i++){
                 ck_assert_msg(fabs(u[i] - expected[i]) < EPS, "wrong item \
                 [%d] %f instead of %f", i,u[i],expected[i]);
            }
    
}
END_TEST


START_TEST(test_matinv)
{

    double c[3][3] = {{ 1, 2, 3}, { 0, 4, 5 }, { 1, 0, 6 }};
    double expected[3][3] = {{1.090909, -0.545455, -0.090909},\
    {0.227273, 0.136364, -0.227273}, {-0.181818, 0.090909, 0.181818}};
    
    int i, j;

            
    matinv ((double*) c, 3, 3);
    
    for (i=0; i<3; i++){
            for (j=0; j<3; j++){
                 ck_assert_msg(fabs(c[i][j] - expected[i][j]) < EPS, "wrong item \
                 [%d][%d] %f instead of %f", i,j,c[i][j],expected[i][j]);
                 }
            }
    
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("lsqadj");
 
    TCase *tc = tcase_create ("lsadj test");
    tcase_add_test(tc, test_matmul);
    tcase_add_test(tc, test_ata);
    tcase_add_test(tc, test_atl);
    tcase_add_test(tc, test_matinv);
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

