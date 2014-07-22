/****************************************************************************

Routine:	       	image_processing.c

Author/Copyright:      	Hans-Gerd Maas

Address:	       	Institute of Geodesy and Photogrammetry
		       	ETH - Hoenggerberg
		       	CH - 8093 Zurich

Creation Date:	       	1988
	
Description:	       	different image processing routines ...
	
Routines contained:    	filter_3:	3*3 filter, reads matrix from filter.par
		       	lowpass_3:	3*3 local average with 9 pointers, fast
		       	lowpass_n:	n*n local average,, fast
		       			computation time independent from n
		       	histogram:	computes histogram
		       	enhance:	enhances gray value spectrum to 0..255,
		      			some extreme gray values are cut off
		       	mark_img:	reads image and pixel coordinate set,
			       		marks the image in a certain (monocomp)
			       		color and writes image

****************************************************************************/

#include <stdio.h>
#include "image_processing.h"

/* filter_3 is a 3 x 3 filter applied to the image
 * filter has to be predefined in the 'filter.par' file in the /parameters folders
 * default filter, if the file is not found or corrupted is [1,1,1; 1,1,1; 1,1,1]/9.
 * Arguments:
 * 8-bit unsigned char image array by pointer *img is an input
 * 8-bit unsigned char image array by pointer *img_lp is an output
 * int imgsize, imx are image size and number of columns (pixels), respectively.
 * in this implementation, boundaries of 1 pixel thickness are untouched and copied 
 * from the original image. the interior is filtered according to the filter.par
 * see ../tests/check_image_processing.c for a couple of useful filters.
 */

void filter_3 (unsigned char *img, unsigned char *img_lp, int imgsize, int imx){

	register unsigned char	*ptr, *ptr1, *ptr2, *ptr3,
		             	    *ptr4, *ptr5, *ptr6,
	                        *ptr7, *ptr8, *ptr9;
	int	       	    end;
	float	       	m[3][3], sum;
	short	       	buf;
	register int	i, j, X, Y, I, J;
	FILE	       	*fp;
	int 			imy; 

	
	/* read filter elements from parameter file */
	fp = fopen ("filter.par","r");
	if (fp == NULL){
	    printf("filter.par was not found, fallback to default lowpass filter \n");
	    for (i=0, sum=9; i<3; i++){
	       for(j=0;j<3; j++){
	          m[i][j] = 1.0/sum; 
	        }
	    }
	} else { 
	      printf("filter.par was found, reading the values \n");  
	      for (i=0, sum=0; i<3; i++){
	          for(j=0; j<3; j++){
		      	fscanf (fp, "%f", &m[i][j]);
		      	// printf("%f", m[i][j]);
		        sum += m[i][j];
		       }
		    }
	    }
	fclose (fp);  
	if (sum == 0) {
	    printf("filter.par is corrupted or empty, fallback to default lowpass filter \n");
	    for (i=0, sum=9; i<3; i++){
	        for (j=0; j<3; j++){
	                m[i][j] = 1.0/sum; 
	        }
	    } 
	}
	
	imy = imgsize/imx;
	
	/* to ensure that the boundaries are original */
	copy_images (img, img_lp, imgsize);
	
	for(Y=0; Y<(imy-2); Y++)  
	{
		for(X=0; X<(imx-2); X++)  
		{
	     buf = 0;
			for(I=0; I<=2; I++)  
			{
				for(J=0; J<=2; J++)  
				{
					buf += (int)( (*(img + X + I + (Y + J)*imx )) * m[I][J]); 
				}
			}
	     // buf/=9;
	     if(buf>255)  buf = 255;
	     if(buf<0)    buf = 0;

	     *(img_lp + X+1 + (Y+1)*imx) = (unsigned char)(buf);	
		}
	}
	
	
	/* old version, 513 is probably for 512 x 512 images, obsolete and 
	*  replaced by the newer version similar to alex_lowpass_3 
	
	
	end = imgsize - 513;
	
	ptr  = img_lp + 513;
	ptr1 = img;				ptr2 = img + 1;			ptr3 = img + 2;
	ptr4 = img + imx;		ptr5 = img + imx + 1;	ptr6 = img + imx + 2;
	ptr7 = img + 2*imx;		ptr8 = img + 2*imx + 1;	ptr9 = img + 2*imx + 2;

	for (i=513; i<end; i++)
	{
		buf = m[0] * *ptr1++  +  m[1] * *ptr2++  +  m[2] * *ptr3++
			+ m[3] * *ptr4++  +  m[4] * *ptr5++  +  m[5] * *ptr6++
			+ m[6] * *ptr7++  +  m[7] * *ptr8++  +  m[8] * *ptr9++;
		buf /= sum;    if (buf > 255)  buf = 255;    if (buf < 8)  buf = 8;
		*ptr++ = buf;
	}
	*/
}


void enhance (unsigned char	*img, int imgsize, int imx ){
	register unsigned char	*ptr;
	unsigned char	       	*end, gmin = 255, gmax = 0, offs;
	float		       	    diff, gain;
	int		       	        i, sum, histo[256];
	
	//void histogram ();
	
	end = img + imgsize;

	histogram (img, histo, imgsize);
	
	for (i=0, sum=0; (i<255)&&(sum<imx); sum += histo[i], i++)  gmin = i;	
	for (i=255, sum=0; (i>0)&&(sum<512); sum+=histo[i], i--)  gmax = i;	
	offs = gmin;  diff = gmax - gmin;  gain = 255 / diff;
	
	for (ptr=img; ptr<end; ptr++)
	{
		if (*ptr < gmin) *ptr = gmin;  else if (*ptr > gmax) *ptr = gmax;
		*ptr = (*ptr - offs) * gain;
		if (*ptr < 8) *ptr = 8;		/* due monocomp colors */
	}
}



/* Apparently enhance is the histogram equalization algorithm but with some
*  constraints that are not clear - why sum is less then imx and then less then 512 
* 
*  New function histeq is an implementation from the 
*  Image Processing in C, 2nd Ed. by Dwayne Phillips, Listing 4.2 
*/
void histeq (unsigned char	*img, int imgsize, int imx ){
	int		       	        X,Y, i, k, imy, histo[256];
	int 			        sum, sum_of_h[256];
    double 					constant;
	
	//void histogram ();
	
	imy = imgsize/imx;

	histogram (img, histo, imgsize);
	
   sum = 0;
   for(i=0; i<256; i++){
      sum += histo[i];
      sum_of_h[i] = sum;
   }
    /* constant = new # of gray levels div by area */
   constant = (float)(255.1)/(float)(imgsize);
   for(Y=0; Y<imy; Y++){
      for(X=0; X<imx; X++){
         k  = *(img + X + Y*imx);
         *(img + X + Y*imx ) = (unsigned char)(sum_of_h[k] * constant);
		} 
	}
}  /* ends histeq */
	



void histogram (unsigned char *img, int *hist, int imgsize){

	int	       	i;
	unsigned char  	*end;
	register unsigned char	*ptr;

	
	for (i=0; i<256; i++)  hist[i] = 0;
	
	end = img + imgsize;
	for (ptr=img; ptr<end; ptr++)
	{
		hist[*ptr]++;
	}
}


/* lowpass_3  is a 2D moving average filter of size 3 x 3 
*  that returns an average of 9 neighbours at the top left corner.
*  Arguments:
*      img, img_lp are the unsigned char array pointers to the original
*      and the low passed images
*      imgsize is the imx * imy the total size of the image
*      imx is the horizontal size of the image
*   
*  See also the more developed version of lowpass_n
* 
*/ 

void lowpass_3 (unsigned char *img, unsigned char *img_lp, int imgsize, int imx){

	register unsigned char	*ptr,*ptr1,*ptr2,*ptr3,*ptr4,
		       		*ptr5,*ptr6,*ptr7,*ptr8,*ptr9;
	short  	       		buf;
	register int   		i, j;
	
	ptr  = img_lp;       // it was img_lp + 513, apparently a bug. 
	ptr1 = img;          // top left corner
	ptr2 = img + 1;      // one to the left
	ptr3 = img + 2;      // to the left
	ptr4 = img + imx;    // one below, mid row left column
	ptr5 = img + imx+1;
	ptr6 = img + imx+2;
	ptr7 = img + 2*imx;
	ptr8 = img + 2*imx+1;
	ptr9 = img + 2*imx+2; // bottom right corner
	
	for (i=0; i<imgsize; i++)
	{
		buf = *ptr5++ + *ptr1++ + *ptr2++ + *ptr3++ + *ptr4++
					  + *ptr6++ + *ptr7++ + *ptr8++ + *ptr9++ ;
		
		*ptr++ = buf/9;
		
	}    
}


void alex_lowpass_3 (unsigned char *img, unsigned char *img_lp, int imgsize, int imx)
{

	int		X, Y;
	int		I, J;
	long	SUM;
	int		imy;
	int		F[3][3]; 
	
	imy = imgsize/imx;
	
	/* 3  X 3 FILTER MASK */
	F[0][0] = 1; F[0][1] = 1; F[0][2] = 1;
	F[1][0] = 1; F[1][1] = 1; F[1][2] = 1;
	F[2][0] = 1; F[2][1] = 1; F[2][2] = 1;
	
	/* to ensure that the boundaries are original */
	
	copy_images (img, img_lp, imgsize);
	
	for(Y=0; Y<(imy-2); Y++)  
	{
		for(X=0; X<(imx-2); X++)  
		{
	     SUM = 0;
			for(I=0; I<=2; I++)  
			{
				for(J=0; J<=2; J++)  
				{
					SUM += (int)( (*(img + X + I + (Y + J)*imx )) * F[I][J]); 
				}
			}
	     SUM/=9;
	     if(SUM>255)  SUM=255;
	     if(SUM<0)    SUM=0;

	     *(img_lp + X+1 + (Y+1)*imx) = (unsigned char)(SUM);	
		}
	}
}

void lowpass_n (int n, unsigned char *img, unsigned char *img_lp, \
                int imgsize, int imx, int imy){

	register unsigned char	*ptrl, *ptrr, *ptrz;
	short  		       	    *buf1, *buf2, buf, *end;
	register short	       	*ptr, *ptr1, *ptr2, *ptr3;
	int    		       	     k, n2, nq;
	register int	       	i;
	
	n2 = 2*n + 1;  nq = n2 * n2;
	
		
	buf1 = (short *) calloc (imgsize, sizeof(short));
	if ( ! buf1)
	{
		printf ("calloc for buf1 --> error \n");
		exit (1);
	}
	
	buf2 = (short *) calloc (imx, sizeof(short));
	if ( ! buf2)
	{
		printf ("calloc for buf2 --> error \n");
		exit (1);
	}


	/* --------------  average over lines  --------------- */
	end = buf1 + imgsize;  buf = 0;
	for (ptrr = img; ptrr < img + n2; ptrr ++)  buf += *ptrr; 
	*(buf1 + n) = buf;
		 
	
	for (ptrl=img, ptr = buf1+n+1; ptr<end; ptrl++, ptr++, ptrr++)
	{
		buf += (*ptrr - *ptrl);  *ptr = buf; 
	}
	
	
	/* -------------  average over columns  -------------- */
	end = buf2 + imx;
	for (ptr1=buf1, ptrz=img_lp+imx*n, ptr3=buf2; ptr3<end;
		 ptr1++, ptrz++, ptr3++)
	{
		for (k=0, ptr2=ptr1; k<n2; k++, ptr2+=imx)  *ptr3 += *ptr2;
		*ptrz = *ptr3/nq;
	}
	for (i=n+1, ptr1=buf1, ptrz=img_lp+imx*(n+1), ptr2=buf1+imx*n2;
		 i<imy-n; i++)
	{
		for (ptr3=buf2; ptr3<end; ptr3++, ptr1++, ptrz++, ptr2++)
		{
			*ptr3 += (*ptr2 - *ptr1);
			*ptrz = *ptr3/nq;
		}
	}


	free (buf1);
	free (buf2);
}




void unsharp_mask (int n, unsigned char *img0, unsigned char *img_lp,\
                   int imgsize, int imx, int imy){
                   
	register unsigned char	*imgum, *ptrl, *ptrr, *ptrz;
	int  		       	*buf1, *buf2, buf, *end;
	register int	       	*ptr, *ptr1, *ptr2, *ptr3;
	int    		       	ii, n2, nq, m;
	register int	       	i;

	n2 = 2*n + 1;  nq = n2 * n2;


	imgum = (unsigned char *) calloc (imgsize, 1);
	if ( ! imgum)
	{
		printf ("calloc for imgum --> error \n");  exit (1);
	}
	
	buf1 = (int *) calloc (imgsize, sizeof(int));
	if ( ! buf1)
	{
		printf ("calloc for buf1 --> error \n");  exit (1);
	}

	buf2 = (int *) calloc (imx, sizeof(int));
	
	if ( ! buf2)
	{
		printf ("calloc for buf2 --> error \n");  exit (1);
	}

	/* set imgum = img0 (so there cannot be written to the original image) */
	for (ptrl=imgum, ptrr=img0; ptrl<(imgum+imgsize); ptrl++, ptrr++)
	{
	  *ptrl = *ptrr;


	}	

	/* cut off high gray values (not in general use !)
	for (ptrz=imgum; ptrz<(imgum+imgsize); ptrz++) if (*ptrz > 160) *ptrz = 160; */




	/* --------------  average over lines  --------------- */

	for (i=0; i<imy; i++)
	{
		ii = i * imx;
		/* first element */
		buf = *(imgum+ii);  *(buf1+ii) = buf * n2;
		
		/* elements 1 ... n */
		for (ptr=buf1+ii+1, ptrr=imgum+ii+2, ptrl=ptrr-1, m=3;
			 ptr<buf1+ii+n+1; ptr++, ptrl+=2, ptrr+=2, m+=2)
		{
			buf += (*ptrl + *ptrr);
			*ptr = buf * n2 / m;
		}
		
		/* elements n+1 ... imx-n */
		for (ptrl=imgum+ii, ptr=buf1+ii+n+1, ptrr=imgum+ii+n2;
			 ptrr<imgum+ii+imx; ptrl++, ptr++, ptrr++)
		{
			buf += (*ptrr - *ptrl);
			*ptr = buf;
		}
		
		/* elements imx-n ... imx */
		for (ptrl=imgum+ii+imx-n2, ptrr=ptrl+1, ptr=buf1+ii+imx-n, m=n2-2;
			 ptr<buf1+ii+imx; ptrl+=2, ptrr+=2, ptr++, m-=2)
		{
			buf -= (*ptrl + *ptrr);
			*ptr = buf * n2 / m;
		}
	}
	


	/* -------------  average over columns  -------------- */

	end = buf2 + imx;

	/* first line */
	for (ptr1=buf1, ptr2=buf2, ptrz=img_lp; ptr2<end; ptr1++, ptr2++, ptrz++)
	{
		*ptr2 = *ptr1;
		*ptrz = *ptr2/n2;
	}
	
	/* lines 1 ... n */
	for (i=1; i<n+1; i++)
	{
		ptr1 = buf1 + (2*i-1)*imx;
		ptr2 = ptr1 + imx;
		ptrz = img_lp + i*imx;
		for (ptr3=buf2; ptr3<end; ptr1++, ptr2++, ptr3++, ptrz++)
		{
			*ptr3 += (*ptr1 + *ptr2);
			*ptrz = n2 * (*ptr3) / nq / (2*i+1);
		}
	}
	
	/* lines n+1 ... imy-n-1 */
	for (i=n+1, ptr1=buf1, ptrz=img_lp+imx*(n+1), ptr2=buf1+imx*n2;
		 i<imy-n; i++)
	{
		for (ptr3=buf2; ptr3<end; ptr3++, ptr1++, ptrz++, ptr2++)
		{
			*ptr3 += (*ptr2 - *ptr1);
			*ptrz = *ptr3/nq;
		}
	}
	
	/* lines imy-n ... imy */
	for (i=n; i>0; i--)
	{
		ptr1 = buf1 + (imy-2*i-1)*imx;
		ptr2 = ptr1 + imx;
		ptrz = img_lp + (imy-i)*imx;
		for (ptr3=buf2; ptr3<end; ptr1++, ptr2++, ptr3++, ptrz++)
		{
			*ptr3 -= (*ptr1 + *ptr2);
			*ptrz = n2 * (*ptr3) / nq / (2*i+1);
		}
	}
	
	free (buf1);
	free (buf2);
	free (imgum);

}


void zoom (unsigned char *img, unsigned char *zoomimg, int xm, int ym, int zf, \
int imgsize, int imx, int imy){
  int          	i0, j0, sx, sy, i1, i2, j1, j2;
  register int	i,j,k,l;


  sx = imx/zf;  sy = imy/zf;  i0 = ym - sy/2;  j0 = xm - sx/2;
  
  /* lines = i, cols = j */
  
  for (i=0; i<sy; i++)  for (j=0; j<sx; j++)
    {
      i1 = i0 + i;  j1 = j0 + j;  i2 = zf*i;  j2 = zf*j;
      for (k=0; k<zf; k++)  for (l=0; l<zf; l++)
	{
	  *(zoomimg + imx*(i2+k) + j2+l) = *(img + imx*i1 + j1);
	}
    }

}


void zoom_new (unsigned char	*img, unsigned char	*zoomimg, int xm, int ym, int zf,\
int zimx, int zimy, int imx){
	int		      	xa, ya;
	register int	       	i;
	register unsigned char	*ptri, *ptrz;
	unsigned char	       	*end;
	
	xa = xm - zimx/(2*zf);
	ya = ym - zimy/(2*zf);
	ptri = img + ya*imx + xa;
	end = zoomimg + zimx*zimy;

	for (ptrz=zoomimg, i=0; ptrz<end; ptrz++)
	{
		*ptrz = *ptri;
		i++;
		if ((i%zimx) == 0)	ptri = img + (ya+(i/(zimx*zf)))*imx + xa;
		if ((i%zf) == 0)	ptri++;
	}
}


/* obsolete function - used with the interlaced video images which were constructed
 * from odd or even lines. if the image was full frames, the function returned the 
 * original image
 */
 	
void split (unsigned char	*img, int field, int imx, int imy, int imgsize){
	register int   		i, j;
	register unsigned char	*ptr;
	unsigned char	       	*end;

	switch (field)
	{
		case 0:  /* frames */
				return;	 break;

		case 1:  /* odd lines */
				for (i=0; i<imy/2; i++)  for (j=0; j<imx; j++)
					*(img + imx*i + j) = *(img + 2*imx*i + j + imx);  break;

		case 2:  /* even lines */
				for (i=0; i<imy/2; i++)  for (j=0; j<imx; j++)
					*(img + imx*i + j) = *(img + 2*imx*i + j);  break;
	}
	
	end = img + imgsize;
	for (ptr=img+imgsize/2; ptr<end; ptr++)  *ptr = 2;
}



void copy_images (unsigned char	*img1, unsigned char *img2, int imgsize){
	register unsigned char 	*ptr1, *ptr2;
	unsigned char	       	*end;


	for (end=img1+imgsize, ptr1=img1, ptr2=img2; ptr1<end; ptr1++, ptr2++)
	*ptr2 = *ptr1;
}



/*------------------------------------------------------------------------
	Subtract mask, Matthias Oswald, Juli 08
  ------------------------------------------------------------------------*/
void subtract_mask (unsigned char	* img, unsigned char	* img_mask, \
unsigned char	* img_new, int imgsize){
	register unsigned char 	*ptr1, *ptr2, *ptr3;
	int i;
	
	for (i=0, ptr1=img, ptr2=img_mask, ptr3=img_new; i<imgsize; ptr1++, ptr2++, ptr3++, i++)
    {
      if (*ptr2 == 0)  *ptr3 = 0;
      else  *ptr3 = *ptr1;
    }
 }


/*
* subtract_img8Bit  is a simple image arithmetic function that subtracts img2 from img1
*  Arguments:
*      img1, img2 are the unsigned char array pointers to the original images
*      img_new is the pointer to the unsigned char array for the resulting image
*      imgsize is the imx * imy the total size of the image
*/
void subtract_img8Bit (unsigned char *img1,unsigned char *img2,unsigned char *img_new, int imgsize) 
{
	register unsigned char 	*ptr1, *ptr2, *ptr3;
	int i;
	
	for (i=0, ptr1=img1, ptr2=img2, ptr3=img_new; i<imgsize; ptr1++, ptr2++, ptr3++, i++)
	{
		if ((*ptr1 - *ptr2) < 0) *ptr3 = 0;
		else  *ptr3 = *ptr1- *ptr2;
	}
 }
