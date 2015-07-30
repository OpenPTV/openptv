/****************************************************************************
 *
 * Image processing routines.
 *
 * Routines contained:    	
 *   filter_3:	3*3 filter, reads matrix from filter.par
 *
 ***************************************************************************/

#include "image_processing.h"

/*  filter_3() performs a 3x3 filtering over an image. The first and last 
    lines are not processed at all, the rest uses wrap-around on the image
    edges. Minimal brightness output in processed pixels is 8.

    Arguments:
    unsigned char *img - original image.
    unsigned char *img_lp - results buffer, same size as original image.
    filter_t filt - the 3x3 matrix to apply to the image.
    control_par *cpar - contains image size parameters.
    
    Returns:
    0 if the filter is bad (all zeros), 1 otherwise.
*/
int filter_3(unsigned char *img, unsigned char *img_lp, filter_t filt, 
    control_par *cpar) {
    
    register unsigned char  *ptr, *ptr1, *ptr2, *ptr3, *ptr4, *ptr5, *ptr6,
        *ptr7, *ptr8, *ptr9;
	int end;
	double sum = 0;
	short buf;
	register int i, j;
    int image_size = cpar->imx * cpar->imy;

    for (i = 0; i < 3; i++)	
        for (j = 0; j < 3; j++)
            sum += filt[i][j];
    if (sum == 0) return 0;
    
    /* start, end etc skip first/last lines and wrap around the edges. */
    end = image_size - cpar->imx - 1;
    
    ptr  = img_lp + cpar->imx + 1;
    ptr1 = img;
    ptr2 = img + 1;
    ptr3 = img + 2;
    
    ptr4 = img + cpar->imx;
    ptr5 = ptr4 + 1;
    ptr6 = ptr4 + 2;
    
    ptr7 = img + 2*cpar->imx;
    ptr8 = ptr7 + 1;
    ptr9 = ptr7 + 2;

    for (i = cpar->imx + 1; i < end; i++) {
        buf = filt[0][0] * *ptr1++ + filt[0][1] * *ptr2++ + filt[0][2] * *ptr3++
            + filt[1][0] * *ptr4++ + filt[1][1] * *ptr5++ + filt[1][2] * *ptr6++
            + filt[2][0] * *ptr7++ + filt[2][1] * *ptr8++ + filt[2][2] * *ptr9++;
        buf /= sum;
        
        if (buf > 255)
            buf = 255;
        if (buf < 8)
            buf = 8;
        
        *ptr++ = buf;
    }
}

