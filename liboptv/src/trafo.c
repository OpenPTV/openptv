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


void pixel_to_metric (double xp, double yp, int imx, int imy, double pix_x, \
double pix_y, double *xc, double *yc, int field){
  switch (field)
    {
    case 1:  yp = 2 * yp + 1;  break;
    case 2:  yp *= 2;  break;
    }
  
  *xc = (xp - imx/2.) * pix_x;
  *yc = (imy/2. - yp) * pix_y;
}




/*  transformation detection pixel coordinates -> geometric coordinates */
/* Arguments - see above for the pixel_to_metric */

void metric_to_pixel (double xc, double yc, int imx, int imy, double pix_x, \
double pix_y, double *xp, double *yp, int field){
  *xp = (xc/pix_x) + imx/2;
  *yp = imy/2 - (yc/pix_y);
  
  switch (field)
    {
    case 1:  *yp = (*yp-1)/2;  break;
    case 2:  *yp /= 2;  break;
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



