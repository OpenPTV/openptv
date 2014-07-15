/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "epi.h"
#include "image_processing.h"

#define EPS 1E-4




START_TEST(test_lowpass_3)
{
        unsigned char *img, *img_lp;
        int imgsize, imx, i, j;
        
        imx = 4;
        imgsize = imx*imx;
        
        /* Allocate the image arrays */
        
        img_lp = (unsigned char *) calloc (imgsize, 1);
        if (! img_lp) {
        	printf ("calloc for img_lp --> error \n");
        	exit (1);
    	}
    	
        img = (unsigned char *) calloc (imgsize, 1);

    	if (! img) {
        	printf ("calloc for img_lp --> error \n");
        	exit (1);
    	}
        
        /* Initialize the image arrays */
        
        for (i=0;i<imgsize;i++){ 
        	img[i] = 128; img_lp[i] = 0; 
        	}     
         
       
        lowpass_3 (img, img_lp, imgsize, imx);
               
            
       ck_assert_msg( fabs(img_lp[0] - 128) < EPS && 
                    fabs(img_lp[7] - 99) < EPS && 
                    fabs(img_lp[15] - 14)  < EPS,
         "\n Expected 128, 99, 14 \n  \
         but found %d %d %d \n", img_lp[0], img_lp[7], img_lp[15] );
}
END_TEST

START_TEST(test_lowpass_n)
{
        unsigned char *img, *img_lp;
        int imgsize, imx, imy, i, j;
        
        imx = imy = 4;
        imgsize = imx*imx;
        
        /* Allocate the image arrays */
        
        img_lp = (unsigned char *) calloc (imgsize, 1);
        if (! img_lp) {
        	printf ("calloc for img_lp --> error \n");
        	exit (1);
    	}
    	
        img = (unsigned char *) calloc (imgsize, 1);

    	if (! img) {
        	printf ("calloc for img_lp --> error \n");
        	exit (1);
    	}
        
        /* Initialize the image arrays */
        
        for (i=0;i<imgsize;i++){ 
        	img[i] = 1; img_lp[i] = 0; 
        	}     
         
       
        lowpass_n (3, img, img_lp, imgsize, imx, imy);
        
        for (i=0; i<imgsize; i++) printf("i img_lp %d %d\n", i, img_lp[i]); 
               
            
        ck_assert_msg( fabs(img_lp[0] - 128) < EPS && 
                    fabs(img_lp[7] - 99) < EPS && 
                    fabs(img_lp[15] - 14)  < EPS,
         "\n Expected 128, 99, 14 \n  \
         but found %d %d %d \n", img_lp[0], img_lp[7], img_lp[15] );
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("image_processing");
    TCase *tc = tcase_create ("image_processing_test");
    tcase_add_test(tc, test_lowpass_3);
    tcase_add_test(tc, test_lowpass_n);
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

