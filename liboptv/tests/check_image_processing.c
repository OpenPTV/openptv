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
        int imgsize, imx, imy, i, j;
        
        imx = imy = 5;
        imgsize = imx*imy;
        
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
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = 128; 
        		//img_lp[i*imx+j] = 0; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 255;
        img[3+5*3] = 255;   

         
        lowpass_3 (img, img_lp, imgsize, imy);
        
        /* print the output */
        printf("--------original ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("--------low-passed ---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img_lp[i*imx+j]);
        	} 
        	printf("\n");
        }
               
            
       ck_assert_msg( img_lp[8] == 142  && 
                      img_lp[12] == 127 && 
                      img_lp[16] == 99 ,
         "\n Expected 142, 127, 99 \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
}
END_TEST


START_TEST(test_lowpass_n)
{
        unsigned char *img, *img_lp;
        int imgsize, imx, imy, i, j;
        
        imx = imy = 5;
        imgsize = imx*imy;
        
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
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = 128; 
        		//img_lp[i*imx+j] = 0; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 255;
        img[3+5*3] = 255;  
         
         
       
        lowpass_n (1, img, img_lp, imgsize, imx, imy);
        
         /* print the output */
        printf("--------original ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("--------low-pass_n ---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img_lp[i*imx+j]);
        	} 
        	printf("\n");
        }
               
            
       ck_assert_msg( img_lp[5] == 99  && 
                      img_lp[10] ==142 && 
                      img_lp[16] == 113 ,
         "\n Expected 99, 142, 113 \n  \
         but found %d %d %d \n", img_lp[5], img_lp[10], img_lp[16] );
}
END_TEST



START_TEST(test_alex_lowpass_3)
{
        unsigned char *img, *img_lp;
        int imgsize, imx, imy, i, j;
        
        imx = imy = 5;
        imgsize = imx*imy;
        
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
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = 128; 
        		//img_lp[i*imx+j] = 0; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 255;
        img[3+5*3] = 255;   

         
        alex_lowpass_3 (img, img_lp, imgsize, imy);
        
        /* print the output */
        printf("--------original ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("--------alex_low-passed ---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img_lp[i*imx+j]);
        	} 
        	printf("\n");
        }
               
            
       ck_assert_msg( img_lp[8] == 113  && 
                      img_lp[12] == 142 && 
                      img_lp[16] == 113 ,
         "\n Expected 113, 142, 113 \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
}
END_TEST

START_TEST(test_histogram)
{
        unsigned char *img;
        int imgsize, imx, imy, i, j;
        int hist[256];
        
        imx = imy = 5;
        imgsize = imx*imy;
        
        /* Allocate the image arrays */
           	
        img = (unsigned char *) calloc (imgsize, 1);

    	if (! img) {
        	printf ("calloc for img_lp --> error \n");
        	exit (1);
    	}
        
        /* Initialize the image arrays */
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = 128; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 255;
        img[3+5*3] = 255;   
         
        histogram (img, hist, imgsize);
        
        // for (i=0; i<256; i++)  printf("i, hist[i] %d %d\n",i,hist[i]);
            
       ck_assert_msg( hist[0]   == 1  && 
                      hist[128] == 22 && 
                      hist[255] == 2 ,
         "\n Expected 1, 22, 2 \n  \
         but found %d %d %d \n", hist[0], hist[128], hist[255]);
}
END_TEST

START_TEST(test_filter_3)
{
        unsigned char *img, *img_lp;
        int imgsize, imx, imy, i, j;
        
        imx = imy = 5;
        imgsize = imx*imy;
        
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
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = 128; 
        		//img_lp[i*imx+j] = 0; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 255;
        img[3+5*3] = 255;   

         
        filter_3 (img, img_lp, imgsize, imx);
        
        /* print the output */
        printf("--------original ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("--------passed filter_3---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img_lp[i*imx+j]);
        	} 
        	printf("\n");
        }
               
            
       ck_assert_msg( img_lp[8] == 113  && 
                      img_lp[12] == 142 && 
                      img_lp[16] == 113 ,
         "\n Expected 113, 142, 113 \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("image_processing");
    TCase *tc = tcase_create ("image_processing_test");
    tcase_add_test(tc, test_lowpass_3);
    tcase_add_test(tc, test_lowpass_n);
    tcase_add_test(tc, test_alex_lowpass_3);
    tcase_add_test(tc, test_histogram);
    tcase_add_test(tc, test_filter_3);
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

