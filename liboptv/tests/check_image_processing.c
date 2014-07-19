/* Unit tests for ray tracing. */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "epi.h"
#include "image_processing.h"

#define EPS 1E-4




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

        /* call the function */ 
        alex_lowpass_3 (img, img_lp, imgsize, imy);
        
        /* print the output */
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("------------------------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img_lp[i*imx+j]);
        	} 
        	printf("\n");
        }
        printf("------------------------------\n");       
            
       // ck_assert_msg( img_lp[8] == 113  && 
//                       img_lp[12] == 142 && 
//                       img_lp[16] == 113 ,
//          "\n Expected 113, 142, 113 \n  \
//          but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
         
         
        lowpass_3 (img, img_lp, imgsize, imy);
        
        /* print the output */
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("------------------------------\n");        
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
        
        imx = imy = 5;
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
        /* it's a part of Lena image, see py_bind/test_lena.py */
        
        img[0] = 235;
        img[1] = 220; 
        img[2] = 207; 
        img[3] = 195;
        img[4] = 180;
        img[5] = 237; 
        img[6] = 230;
        img[7] = 209;
        img[8] = 197;
        img[9] = 176;
        img[10] = 238;
        img[11] = 222;
        img[12] = 204;
        img[13] = 191;
        img[14] = 177;
        img[15] = 227;
        img[16] = 212;
        img[17] = 199;
        img[18] = 189;
        img[19] = 177;
        img[20] = 225; 
        img[21] = 215;
        img[22] = 197;
        img[23] = 179;
        img[24] = 173;
         
         
       
        lowpass_n (2, img, img_lp, imgsize, imx, imy);
        
        for (i=0; i<imgsize; i++) printf("i img img_lp %d %d %d \n", i, img[i], img_lp[i]); 
               
            
        ck_assert_msg( fabs(img_lp[10] - 146) < EPS && 
                    fabs(img_lp[12] - 188) < EPS && 
                    fabs(img_lp[14] - 185)  < EPS,
         "\n Expected 146, 188, 185 \n  \
         but found %d %d %d \n", img_lp[0], img_lp[7], img_lp[15] );
}
END_TEST

Suite* fb_suite(void) {
    Suite *s = suite_create ("image_processing");
    TCase *tc = tcase_create ("image_processing_test");
    tcase_add_test(tc, test_alex_lowpass_3);
    // tcase_add_test(tc, test_lowpass_3);
    // tcase_add_test(tc, test_lowpass_n);
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

