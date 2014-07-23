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
         
         
       
        lowpass_n (1, img, img_lp, imgsize, imx);
        
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



START_TEST(test_lowpass_3_cb)
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

         
        lowpass_3_cb (img, img_lp, imgsize, imy);
        
        /* print the output */
        printf("--------original ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        printf("--------low-passed_cb ---------------\n");        
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

START_TEST(test_enhance)
{
        unsigned char *img;
        int imgsize, imx, imy, i, j;
        int hist[256];
        
        imx = 5;
        imy = 7;
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
        		img[i*imx+j] = (i+1)*(j+1)*10; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 181;
        img[3+5*3] = 255;   
        
        /* print the output */
        
       printf("--------before enhance ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        
        
         
        enhance(img, imgsize, imx);
        
        printf("--------after enhance ---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        }
        
        // for (i=0; i<256; i++)  printf("i, hist[i] %d %d\n",i,hist[i]);
            
        ck_assert_msg(img[8] == 255  && 
                      img[12] == 8 && 
                      img[16] == 255 ,
         "\n Expected 126 84 126 \n  \
         but found %d %d %d \n", img[8], img[12], img[16] );
         
         
         
         /* Now let's try histeq */
         /* Initialize the image arrays */
         
         imx = 6;
         imy = 5;	
         
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = (i+1)*(j+1)*5; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 181;
        img[imx*imy-1] = 255;   
        
        /* print the output */
        
       printf("--------before histeq ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        
        
         
        histeq(img, imgsize, imx);
        
        printf("--------after histeq ---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        }
        
        // for (i=0; i<256; i++)  printf("i, hist[i] %d %d\n",i,hist[i]);
            
        ck_assert_msg(img[2] == 29  && 
                      img[7] == 51 && 
                      img[24] == 65 &&
                      img[imgsize-1] == 255,
         "\n Expected 29 51 65 255 \n  \
         but found %d %d %d %d \n", img[2], img[7], img[24], img[imgsize-1] );
         
         
}
END_TEST

START_TEST(test_filter_3)
{
        unsigned char *img, *img_lp;
        int imgsize, imx, imy, i, j;
        
        int	       	    end;
	    float	       	m[3][3], sum;
	    FILE	       	*fp;
	    
	    

	    m[0][0] = 0; m[0][1] = 1; m[0][2] = 0;
	    m[1][0] = 1; m[1][1] = 2; m[1][2] = 1;
	    m[2][0] = 0; m[2][1] = 1; m[2][2] = 0;
	    sum = 6.0;
	    
	    
	    /* write filter elements to the parameter file */
	    fp = fopen ("filter.par","w");
	    if (fp != NULL){ 
	      for (i=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fprintf (fp, "%4.3f\t", m[i][j]/sum);
		       }
		    }
	   fclose (fp); 
	   } 
        
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
               
            
       ck_assert_msg( img_lp[8] == 126  && 
                      img_lp[12] == 84 && 
                      img_lp[16] == 126 ,
         "\n Expected 126 84 126 \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
         
         
         /* another filter */
        m[0][0] = 1; m[0][1] = 1; m[0][2] = 1;
	    m[1][0] = 1; m[1][1] = 2; m[1][2] = 1;
	    m[2][0] = 1; m[2][1] = 1; m[2][2] = 1;
	    sum = 10.0;
	    
	    
	    /* write filter elements to the parameter file */
	    fp = fopen ("filter.par","w");
	    if (fp != NULL){ 
	      for (i=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fprintf (fp, "%4.3f\t", m[i][j]/sum);
		       }
		    }
	   fclose (fp); 
	   } 
       
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
               
            
       ck_assert_msg( img_lp[8] == 109  && 
                      img_lp[12] == 122 && 
                      img_lp[16] == 109 ,
         "\n Expected 109 122 109 \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
         
         /* another filter */
        m[0][0] = 1; m[0][1] = 2; m[0][2] = 1;
	    m[1][0] = 2; m[1][1] = 4; m[1][2] = 2;
	    m[2][0] = 1; m[2][1] = 2; m[2][2] = 1;
	    sum = 16.0;
	    
	    
	    /* write filter elements to the parameter file */
	    fp = fopen ("filter.par","w");
	    if (fp != NULL){ 
	      for (i=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fprintf (fp, "%4.3f\t", m[i][j]/sum);
		       }
		    }
	   fclose (fp); 
	   } 
       
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
               
            
       ck_assert_msg( img_lp[8] == 117  && 
                      img_lp[12] == 108 && 
                      img_lp[16] == 117 ,
         "\n Expected 117 108 117  \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
         
         
         
           /* low pass  */
        m[0][0] = 1; m[0][1] = 1; m[0][2] = 1;
	    m[1][0] = 1; m[1][1] = 1; m[1][2] = 1;
	    m[2][0] = 1; m[2][1] = 1; m[2][2] = 1;
	    sum = 9.0;
	    
	    
	    /* write filter elements to the parameter file */
	    fp = fopen ("filter.par","w");
	    if (fp != NULL){ 
	      for (i=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fprintf (fp,"%4.3f\t", m[i][j]/sum);
		       }
		    }
	   fclose (fp); 
	   } 
       
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
               
            
       ck_assert_msg( img_lp[8] == 	112  && 
                      img_lp[12] == 140 && 
                      img_lp[16] == 112 ,
         "\n Expected 112 140 112  \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] ); 
         
         
          /* corrupted */
        m[0][0] = 0; m[0][1] = 0; m[0][2] = 0;
	    m[1][0] = 0; m[1][1] = 0; m[1][2] = 0;
	    m[2][0] = 0; m[2][1] = 0; m[2][2] = 0;
	    sum = 16.0;
	    
	    
	    /* write filter elements to the parameter file */
	    fp = fopen ("filter.par","w");
	    if (fp != NULL){ 
	      for (i=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fprintf (fp, "%4.3f\t", m[i][j]/sum);
		       }
		    }
	   fclose (fp); 
	   } 
       
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
               
            
       ck_assert_msg( img_lp[8] == 112  && 
                      img_lp[12] == 140 && 
                      img_lp[16] == 112 ,
         "\n Expected 112 140 112  \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[16] );
         
         
         /* let's add few high pass filter tests */
        m[0][0] = -1; m[0][1] = -1; m[0][2] = -1;
	    m[1][0] = -1; m[1][1] = 9; m[1][2] =  -1;
	    m[2][0] = -1; m[2][1] = -1; m[2][2] = -1;
	    sum = 1.0;
	    
	    
	    /* write filter elements to the parameter file */
	    fp = fopen ("filter.par","w");
	    if (fp != NULL){ 
	      for (i=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fprintf (fp, "%4.3f\t", m[i][j]/sum);
		       }
		    }
	   fclose (fp); 
	   } 
       
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
               
            
       ck_assert_msg( img_lp[8] == 255  && 
                      img_lp[12] == 0 && 
                      img_lp[17] == 129 ,
         "\n Expected 255 0 129 \n  \
         but found %d %d %d \n", img_lp[8], img_lp[12], img_lp[17] );
         
         /* another filter */
        m[0][0] = 1; m[0][1] = 2; m[0][2] = 1;
	    m[1][0] = 2; m[1][1] = 4; m[1][2] = 2;
	    m[2][0] = 1; m[2][1] = 2; m[2][2] = 1;
	    sum = 16.0;
         
         
         
}
END_TEST


START_TEST(test_highpass)
{
        unsigned char *img, *img_hp;
        int imgsize, imx, imy, i, j, dim_lp, filter_hp;
        int hist[256];
        
        imx = 5;
        imy = 5;
        imgsize = imx*imy;
        
        /* Allocate the image arrays */
           	
        img = (unsigned char *) calloc (imgsize, 1);

    	if (! img) {
        	printf ("calloc for img --> error \n");
        	exit (1);
    	}
    	
    	img_hp = (unsigned char *) calloc (imgsize, 1);

    	if (! img_hp) {
        	printf ("calloc for img_hp --> error \n");
        	exit (1);
    	}
        
        /* Initialize the image arrays */
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		img[i*imx+j] = 127; //(i+1)*(j+1)*10; 
        	} 
        } 
        img[2+5*2] = 0;
        img[1+5*1] = 181;
        img[3+5*3] = 255;   
        
        /* print the output */
        
       printf("--------before highpass ---------------\n"); 
       for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img[i*imx+j]);
        	} 
        	printf("\n");
        } 
        
        dim_lp = 1;
        filter_hp = 1;
         
        highpass(img, img_hp, dim_lp, filter_hp, imgsize, imx);
        
        printf("--------after highpass ---------------\n");        
        for (i=0;i<imy;i++){ 
            for(j=0;j<imx;j++){
        		printf("%d\t", img_hp[i*imx+j]);
        	} 
        	printf("\n");
        }
        
        // for (i=0; i<256; i++)  printf("i, hist[i] %d %d\n",i,hist[i]);
            
        ck_assert_msg(img_hp[8] == 255  && 
                      img_hp[12] == 8 && 
                      img_hp[16] == 255 ,
         "\n Expected 126 84 126 \n  \
         but found %d %d %d \n", img_hp[8], img_hp[12], img_hp[16] );
 
         
}
END_TEST


Suite* fb_suite(void) {
    Suite *s = suite_create ("image_processing");
    TCase *tc = tcase_create ("image_processing_test");
    tcase_add_test(tc, test_lowpass_3);
    tcase_add_test(tc, test_lowpass_n);
    tcase_add_test(tc, test_lowpass_3_cb);
    tcase_add_test(tc, test_histogram);
    tcase_add_test(tc, test_filter_3);
    tcase_add_test(tc, test_enhance);
    tcase_add_test(tc, test_highpass);
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

