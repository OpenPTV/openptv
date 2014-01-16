/****************************************************************************

Routine:	       		trafo.c

Author/Copyright:		Hans-Gerd Maas

Address:	       		Institute of Geodesy and Photogrammetry
		       	        ETH - Hoenggerberg
			       	CH - 8093 Zurich

Creation Date:			25.5.88
	
Description:			diverse transformations
	
Routines contained:		pix_crd, crd_pix, affin_trafo, affin_retour

****************************************************************************/

#include "trafo.h"

void old_pixel_to_metric();
void old_metric_to_pixel();

/* pixel_to_metric converts pixel coordinates to metric coordinates
Arguments:
	xp,yp (double) pixel coordinates in pixels
	(xc,yc) (double *) metric coordinates in [mm]
	imx,imy (int) image size in pixels
	pix_x, pix_y (double) size of the pixel of the sensor, in [mm]  
    field (flag [int], 0 is frame, 1 is for odd or 2 is for even fields) 
    Note: field or chfield in the parameters is not used anymore (no interlaced cameras) 
    and it is kept only for backward compatibility. 
*/

/*  transformation detection pixel coordinates -> geometric coordinates */
void old_pixel_to_metric (double * x_metric
		       , double * y_metric
		       , double x_pixel
		       , double y_pixel
		       , int im_size_x
		       , int im_size_y
		       , double pix_size_x
		       , double pix_size_y
		       , int y_remap_mode){
  
  switch (y_remap_mode){
  case NO_REMAP:
    break;

  case DOUBLED_PLUS_ONE:  
    y_pixel = 2. * y_pixel + 1.;  
    break;
    
  case DOUBLED:  
    y_pixel *= 2.;  
    break;  
  }
  
  *x_metric = (x_pixel - ((double)im_size_x) / 2.) * pix_size_x;
  *y_metric = ( ((double) im_size_y)/2. - y_pixel) * pix_size_y;

}

/*  wraps previous one, parameters are read directly from control_par* structure */
void pixel_to_metric(double * x_metric
				 , double * y_metric
				 , double x_pixel
				 , double y_pixel
				 , control_par* parameters				 
				 ){
  old_pixel_to_metric(x_metric
		  , y_metric
		  , x_pixel
		  , y_pixel
		  , parameters->imx
		  , parameters->imy
		  , parameters->pix_x
		  , parameters->pix_y
		  , parameters->chfield);


}

/* wrap metric_to_pixel */
void metric_to_pixel(double * x_pixel
				 , double * y_pixel
				 , double x_metric
				 , double y_metric
				 , control_par* parameters				 
				 ){
  old_metric_to_pixel(x_pixel
		  , y_pixel
		  , x_metric
		  , y_metric
		  , parameters->imx
		  , parameters->imy
		  , parameters->pix_x
		  , parameters->pix_y
		  , parameters->chfield);
}


/*  transformation detection geometric coordinates -> pixel coordinates */
void old_metric_to_pixel (double * x_pixel
		      , double * y_pixel
		      , double x_metric
		      , double y_metric
		      , int im_size_x
		      , int im_size_y
		      , double pix_size_x
		      , double pix_size_y
		      , int y_remap_mode){
  
  
  *x_pixel = ( x_metric / pix_size_x ) + ( (double) im_size_x)/2.;
  *y_pixel = ((double)im_size_y) /2. - (y_metric / pix_size_y);

  switch (y_remap_mode){
  case NO_REMAP:
    break;

  case DOUBLED_PLUS_ONE:  
    *y_pixel = (*y_pixel - 1.)/2.;  
    break;
    
  case DOUBLED:  
    *y_pixel /= 2.;
    break;  
  }
}

/*  transformation with Brown + affine  */
void distort_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1){
  double		r;
  
  
  r = sqrt (x*x + y*y);
  if (r != 0)
    {
      x += x * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
	+ ap.p1 * (r*r + 2*x*x) + 2*ap.p2*x*y;
      
      y += y * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
	+ ap.p2 * (r*r + 2*y*y) + 2*ap.p1*x*y;
      
      *x1 = ap.scx * x - sin(ap.she) * y;  *y1 = cos(ap.she) * y;
    }
}



/*  correct crd to geo with Brown + affine  */
void correct_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1){

  double  r, xq, yq;
	

  r = sqrt (x*x + y*y);
  if (r != 0)
    {
      xq = (x + y*sin(ap.she))/ap.scx
	- x * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
	- ap.p1 * (r*r + 2*x*x) - 2*ap.p2*x*y;
      yq = y/cos(ap.she)
	- y * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
	- ap.p2 * (r*r + 2*y*y) - 2*ap.p1*x*y;
    }
  r = sqrt (xq*xq + yq*yq);		/* one iteration */
  if (r != 0)
    {
      *x1 = (x + yq*sin(ap.she))/ap.scx
	- xq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
	- ap.p1 * (r*r + 2*xq*xq) - 2*ap.p2*xq*yq;
      *y1 = y/cos(ap.she)
	- yq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
	- ap.p2 * (r*r + 2*yq*yq) - 2*ap.p1*xq*yq;
    }
}



