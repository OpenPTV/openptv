/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "lsqadj.h"

#define EPS 1E-5


START_TEST(test_norm_cross)
{

    double n[3];

    // test simple cross-product normalized to unity

    double a[] = {1.0, 0.0, 0.0};
    double b[] = {0.0, 2.0, 0.0};

    norm_cross(a,b,n);
    fail_unless( (n[0] == 0.0) && (n[1] == 0.0) && (n[2] == 1.0));


    // test negative values in the output

    norm_cross(b,a,n);
    // fail_unless( (n[0] == 0.0) && (n[1] == 0.0) && (n[2] == -1.0));


    ck_assert_msg( fabs(n[0] - 0.0) < EPS && 
                   fabs(n[1] - 0.0) < EPS && 
                   fabs(n[2] - -1.0)  < EPS,
             "Was expecting n to be 0., 0., -1. but found %f %f %f\n", n[0],n[1],n[2]);
         

    // our norm_cross had a bug when multiplying the parallel vectors
    // it was returning nan instead of 0.0
    // fixed Aug. 3, 2013, see in ray_tracing.c

    norm_cross(a,a,n);
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
    tcase_add_test(tc, test_norm_cross);
    tcase_add_test(tc, test_dot);
    tcase_add_test(tc, test_modu);
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

