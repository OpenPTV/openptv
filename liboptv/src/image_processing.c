/****************************************************************************
 *
 * Image processing routines.
 *
 * Routines contained:    	
 *   filter_3:	3*3 filter, reads matrix from filter.par
 *
 ***************************************************************************/

#include <stdlib.h>
#include "image_processing.h"

/*  This would be a function, only the original writer of these filters put a 
    lot of emphasis on speed and used 'register' variables. this probably does 
    nothing on a modern compiler, esp. in 32 bit where there are simply not 
    enough registers, but I honor the intent by inlining this function. */
#define setup_filter_pointers(line) \
    ptr  = img_lp + (line) + 1;\
\
    ptr1 = img;\
    ptr2 = img + 1;\
    ptr3 = img + 2;\
\
    ptr4 = img + (line);\
    ptr5 = ptr4 + 1;\
    ptr6 = ptr4 + 2;\
\
    ptr7 = img + 2*(line);\
    ptr8 = ptr7 + 1;\
    ptr9 = ptr7 + 2;\


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
	unsigned short buf;
	register int i, j;
    int image_size = cpar->imx * cpar->imy;

    for (i = 0; i < 3; i++)	
        for (j = 0; j < 3; j++)
            sum += filt[i][j];
    if (sum == 0) return 0;
    
    /* start, end etc skip first/last lines and wrap around the edges. */
    end = image_size - cpar->imx - 1;
    
    setup_filter_pointers(cpar->imx)
    for (i = cpar->imx + 1; i < end; i++) {
        buf = filt[0][0] * *ptr1++ + filt[0][1] * *ptr2++ + filt[0][2] * *ptr3++
            + filt[1][0] * *ptr4++ + filt[1][1] * *ptr5++ + filt[1][2] * *ptr6++
            + filt[2][0] * *ptr7++ + filt[2][1] * *ptr8++ + filt[2][2] * *ptr9++;
        buf /= sum;
        
        if (buf > 255)
            buf = 255;
        if (buf < 8)
            buf = 8;
        
        *ptr++ = (unsigned char) buf;
    }
    return 1;
}


/*  This is a reduced version with a constant meadian filter (average of all 9
    pixels in filter range). It also does not enforce minimal brightness.
    
    Arguments:
    unsigned char *img - original image.
    unsigned char *img_lp - results buffer, same size as original image.
    control_par *cpar - contains image size parameters.
*/
void lowpass_3(unsigned char *img, unsigned char *img_lp, control_par *cpar) {
    register unsigned char  *ptr, *ptr1, *ptr2, *ptr3, *ptr4, *ptr5, *ptr6,
        *ptr7, *ptr8, *ptr9;
    int end;
    short buf;
    register int i;
    int image_size = cpar->imx * cpar->imy;
    
    /* start, end etc skip first/last lines and wrap around the edges. */
    end = image_size - cpar->imx - 1;
    
    setup_filter_pointers(cpar->imx)
    for (i = cpar->imx + 1; i < end; i++) {
        buf = *ptr5++ + *ptr1++ + *ptr2++ + *ptr3++ + *ptr4++
                      + *ptr6++ + *ptr7++ + *ptr8++ + *ptr9++;
        *ptr++ = buf/9;
    }
}


/*  fast_box_blur() performs a box blur of an image using a given kernel size.
    It is equivalent to using an all-ones kernel of the given size in a 
    function like filter_3 (but adjusted to size). However, this algorithm runs
    in linear filter-size time instead of quadratic (it's still quadratic in 
    image size).
    
    Arguments:
    int filt_span - how many pixels to take for the average on each side. The
        equivalent filter kernel is of side 2*filt_size + 1.
    unsigned char *src - points to the source image.
    unsigned char *dest - points to same-size memory block allocated for the 
        result.
    control_par *cpar - contains image size parameters.
    
    Returns:
    0 on failure (due to memory allocation error), 1 on success.
    
    References:
    [1] http://www.filtermeister.com/tutorials/blur01.html
    [2] http://www.gamasutra.com/view/feature/131511/four_tricks_for_fast_blurring_in_.php
*/
int fast_box_blur(int filt_span, unsigned char *src, unsigned char *dest, 
    control_par *cpar) 
{
    register unsigned char  *ptrl, *ptrr, *ptrz;
    register int *ptr, *ptr1, *ptr2, *ptr3;
    int *row_accum, *col_accum, accum, *end;
    int row_start, n, nq, m;
    register int i;
    int image_size = cpar->imx * cpar->imy;
    
    n = 2*filt_span + 1;
    nq = n*n;
    
    if ((row_accum = (int *)calloc(image_size, sizeof(int))) == NULL) return 0;
    if ((col_accum = (int *)calloc(cpar->imx, sizeof(int))) == NULL) return 0;
    
    /* Sum over lines first [1]: */
    for (i = 0; i < cpar->imy; i++) {
        row_start = i * cpar->imx;
        
        /* first element has no filter around him */
        accum = *(src + row_start);
        *(row_accum + row_start) = accum * n;
        
        /* Elements 1..filt_span have a growing filter, as much as fits.
           Each iteration increases the filter symmetrically, so 2 new elements 
           are taken.
        */
        for (ptr = row_accum + row_start + 1, 
            ptrr = src + row_start + 2, ptrl = ptrr - 1, m = 3;
            ptr < row_accum + row_start + 1 + filt_span;
            ptr++, ptrl+=2, ptrr+=2, m+=2) 
        {    
            accum += (*ptrl + *ptrr);
            *ptr = accum * n / m; /* So small filters have same weight as
                                         the largest size */
        }
        
        /* Middle elements, having a constant-size filter. The sum is obtained
           by adding the coming element and dropping the leaving element, in a 
           sliding window fashion. 
        */
        for (ptr = row_accum + row_start + filt_span + 1,
            ptrl = src + row_start, ptrr = src + row_start + n;
            ptrr < src + row_start + cpar->imx;
            ptrl++, ptr++, ptrr++)
        {
            accum += (*ptrr - *ptrl);
            *ptr = accum;
        }
        
        /* last elements in line treated like first ones, mutatis mutandis */
        for (ptrl = src + row_start + cpar->imx - n,
            ptrr = ptrl + 1, m = n - 2,
            ptr = row_accum + row_start + cpar->imx - filt_span;
            ptr < row_accum + row_start + cpar->imx;
            ptrl += 2, ptrr += 2, ptr++, m-=2)
        {
            accum -= (*ptrl + *ptrr);
            *ptr = accum * n / m;
        }
    }
    
    /* Sum over columns: */
    
    end = col_accum + cpar->imx;
    
    /* first line */
    for (ptr1 = row_accum, ptr2 = col_accum, ptrz = dest; 
        ptr2 < end; ptr1++, ptr2++, ptrz++)
    {
       *ptr2 = *ptr1;
       *ptrz = *ptr2/n;
    }
    
    /* Sum vertically the accumulated row values for lines 1 ... filt_span */
    for (i = 1; i <= filt_span; i++) {
        ptr1 = row_accum + (2*i - 1)*cpar->imx;
        ptr2 = ptr1 + cpar->imx;
        ptrz = dest + i*cpar->imx;
        
        for (ptr3 = col_accum; ptr3 < end; ptr1++, ptr2++, ptr3++, ptrz++) {
            *ptr3 += (*ptr1 + *ptr2);
            *ptrz = n * (*ptr3) / nq / (2*i + 1);
        }
    }
    
    /* Middle lines with filter-size lines around them */
    for (i = filt_span + 1, ptr1 = row_accum, 
        ptrz = dest + cpar->imx*(filt_span + 1),
        ptr2 = row_accum + cpar->imx*n; i < cpar->imy - filt_span; i++)
    {
        for (ptr3 = col_accum; ptr3 < end; ptr3++, ptr1++, ptrz++, ptr2++) {
           *ptr3 += (*ptr2 - *ptr1);
           *ptrz = *ptr3/nq;
        }
    }
    
    /* Last lines, similarly to first lines */
    for (i = filt_span; i > 0; i--) {
        ptr1 = row_accum + (cpar->imy - 2*i - 1)*cpar->imx;
        ptr2 = ptr1 + cpar->imx;
        ptrz = dest + (cpar->imy-i)*cpar->imx;
        
        for (ptr3 = col_accum; ptr3 < end; ptr1++, ptr2++, ptr3++, ptrz++) {
            *ptr3 -= (*ptr1 + *ptr2);
            *ptrz = n * (*ptr3) / nq / (2*i+1);
        }
    }
    
    /* ``dest`` now contains result. Finish up. */
    free(row_accum);
    free(col_accum);
    
    return 1;
}


/*  split() crams into the first half of a given image either its even or odd 
    lines. Used with interlaced cameras, a mostly obsolete device.
    The lower half of the image is set to the number 2.
    
    Arguments:
    unsigned char *img - the image to modify. Both input and output.
    int half_selector - 0 to do nothing, 1 to take odd rows, 2 for even rows
    control_par *cpar - contains image size parameters.
*/
void split(unsigned char *img, int half_selector, control_par *cpar) {
    register int row, col;
    register unsigned char *ptr;
    unsigned char *end;
    int image_size = cpar->imx * cpar->imy;
    int cond_offs = (half_selector % 2) ? (cpar->imx) : (0);
    
    if (half_selector == 0)
        return;
    
    for (row = 0; row < cpar->imy/2; row++)
        for (col = 0; col < cpar->imx; col++)
            img[row*cpar->imx + col] = img[2*row*cpar->imx + cond_offs + col];
    
    /* Erase lower half with magic 2 */
    end = img + image_size;
    for (ptr = img + image_size/2; ptr < end; ptr++)
        *ptr = 2;
}


/*
* subtract_img  is a simple image arithmetic function that subtracts img2 from img1
*  Arguments:
*      img1, img2 are the unsigned char array pointers to the original images
*      img_new is the pointer to the unsigned char array for the resulting image
*      control_par *cpar - contains image size parameters.
*/
void subtract_img (unsigned char *img1,unsigned char *img2,unsigned char *img_new, control_par *cpar) 
{
	register unsigned char 	*ptr1, *ptr2, *ptr3;
	int i;
	int image_size = cpar->imx * cpar->imy;
	
	for (i=0, ptr1=img1, ptr2=img2, ptr3=img_new; i<image_size; ptr1++, ptr2++, ptr3++, i++)
	{
		if ((*ptr1 - *ptr2) < 0) *ptr3 = 0;
		else  *ptr3 = *ptr1- *ptr2;
	}
}


/*
* Subtract_mask, by Matthias Oswald, Juli 08
* subtract_mask compares img with img_mask and creates a masked image img_new
* pixels that are equal to zero in the img_mask
* are overwritten with a default value (=0) in img_new
*  Arguments:
*      img  is the unsigned char array pointers to the original image
%      img  is the unsigned char array pointers to the mask (0 == mask out)
*      img_new is the pointer to the unsigned char array for the resulting image
*      control_par *cpar - contains image size parameters.
*/
void subtract_mask (unsigned char *img, unsigned char *img_mask, unsigned char *img_new, control_par *cpar)
{
	register unsigned char 	*ptr1, *ptr2, *ptr3;
	int i;
	int image_size = cpar->imx * cpar->imy;
	
	for (i=0, ptr1=img, ptr2=img_mask, ptr3=img_new; i<image_size; ptr1++, ptr2++, ptr3++, i++)
    {
      if (*ptr2 == 0)  *ptr3 = 0;
      else  *ptr3 = *ptr1;
    }
 }
 
/*
*  copy_images  is a simple image arithmetic function that copies img1 into img2
*  Arguments:
*      img1, img2 are the unsigned char array pointers 
*      control_par *cpar - contains image size parameters.
*/
void copy_images (unsigned char	*img1, unsigned char *img2, control_par *cpar)
{
	register unsigned char 	*ptr1, *ptr2;
	unsigned char	       	*end;
	int image_size = cpar->imx * cpar->imy;
	
	for (end=img1+image_size, ptr1=img1, ptr2=img2; ptr1<end; ptr1++, ptr2++)
	*ptr2 = *ptr1;
}

/* histogram computes a histogram from the image and returns it to hist
*  Arguments:
*      img  is the unsigned char array pointer to the image array, 8 bit 
*      control_par *cpar - contains image size parameters.
*/
void histogram (unsigned char *img, int *hist, control_par *cpar){

	int	       	i;
	unsigned char  	*end;
	register unsigned char	*ptr;
	int imgsize = cpar->imx * cpar->imy;

	//write defaults into histgramm array
	for (i=0; i<256; i++)  hist[i] = 0;
	
	end = img + imgsize;
	for (ptr=img; ptr<end; ptr++)
	{
		hist[*ptr]++;
	}
}

/* Apparently enhance is the histogram equalization algorithm but with some
*  constraints that are not clear - why sum is less then imx and then less then 512 
* 
*  New function histeq is an implementation from the 
*  Image Processing in C, 2nd Ed. by Dwayne Phillips, Listing 4.2 
*  Arguments:
*      img  is the unsigned char array pointer to the image array, 8 bit 
*      control_par *cpar - contains image size parameters.
*/
void histeq (unsigned char	*img, control_par *cpar)
{
	int		       	        i, j, k, histo[256];
	int 			        sum, sum_of_h[256];
    double 					constant;
    int imgsize = cpar->imx * cpar->imy;
	
	histogram (img, histo, cpar);
	
	sum = 0;
	for(i=0; i<256; i++)
	{
		sum += histo[i];
		sum_of_h[i] = sum;
	}
    /* constant = new # of gray levels div by area */
	constant = (float)(255.1)/(float)(imgsize);
	for(j=0; j<cpar->imy; j++)
	{
		for(i=0; i<cpar->imx; i++)
		{
			k  = *(img + i + j*cpar->imx);
			*(img + i + j*cpar->imx ) = (unsigned char)(sum_of_h[k] * constant);
		} 
	}
} 
