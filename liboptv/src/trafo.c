/****************************************************************************
Based on initial code from Hans-Gerd Maas, creation date 25.5.88
	
Description: transformations between 2D coordinate systems: pixel coordinates
on the image plane; metric representation where max x,y are sensor width/height
in mm; and flat-image coordinates, where metric coordinates are cleaned of
distortion and sensor shift so they may be used for ray tracing in a simple
pinhole-camera model.
	
References:
[1] Dracos, Th. (ed.), Three-Dimensional Velocity and Vorticity Measurements
    and Image Analysis Techniques. Kluwer Academic Publishers, 1996.
    
****************************************************************************/

#include "trafo.h"

/* for the correction routine: */
#include <math.h>

/*  old_pixel_to_metric() converts pixel coordinates to metric coordinates
    
    Arguments:
    double *x_metric, *y_metric - output metric coordinates.
    double x_pixel, y_pixel - input pixel coordinates.
    int im_size_x, im_size_y - size in pixels of the corresponding image 
        dimensions.
    double pix_size_x, pix_size_y - metric size of each pixel on the sensor 
        plane.
    int y_remap_mode - for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
*/
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

/*  pixel_to_metric() converts pixel coordinates to metric coordinates, given
    a modern configurattion.
    
    Arguments:
    double *x_metric, *y_metric - output metric coordinates.
    double x_pixel, y_pixel - input pixel coordinates.
    control_par* parameters - control structure holding image and pixel sizes.
    int y_remap_mode - for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
*/
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

/*  old_metric_to_pixel() converts metric coordinates to pixel coordinates
    
    Arguments:
    double *x_pixel, *y_pixel - input pixel coordinates.
    double x_metric, y_metric - output metric coordinates.
    int im_size_x, im_size_y - size in pixels of the corresponding image 
        dimensions.
    double pix_size_x, pix_size_y - metric size of each pixel on the sensor 
        plane.
    int y_remap_mode - for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
*/
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

/*  metric_to_pixel() converts metric coordinates to pixel coordinates, given
    a modern configurattion.
    
    Arguments:
    double *x_pixel, *y_pixel - input pixel coordinates.
    double x_metric, y_metric - output metric coordinates.
    control_par* parameters - control structure holding image and pixel sizes.
    int y_remap_mode - for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
*/
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


/*  distort_brown_affin() Calculates the real image-plane position of a point
    seen by a camera with distortions (Brown distortions and scale/shear
    affine transforms), from a point that would be seen by a simple pinhole
    camera.
    
    Arguments:
    double x,y - undistorted metric coordinates.
    ap_52 ap - distortion parameters struct.
    double *x1, *y1 - output metric distorted parameters.
*/
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

/*  correct distorted to flat-image coordinates, see correct_brown_affine_exact(),
    this one is the same except it ensures only one iteration for backward 
    compatibility.
*/
void correct_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1)
{
    correct_brown_affine_exact(x, y, ap, x1, y1, 100000);
}

/*  correct_brown_affine_exact() attempts to iteratively solve the inverse
    problem of what flat-image coordinate yielded the given distorted 
    coordinates.
    
    Arguments:
    double x, y - input metric shifted real-image coordinates.
    ap_52 ap - distortion parameters used in the distorting step.
    double *x1, *y1 - output metric shifted flat-image coordinates. Still needs
        unshifting to get pinhole-equivalent coordinates.
    tol - stop if the relative improvement in position between iterations is 
        less than this value.
*/
void correct_brown_affine_exact(double x, double y, ap_52 ap, 
    double *x1, double *y1, double tol)
{
    double  r, rq, xq, yq;
    int itnum = 0;
  
    if ((x == 0) && (y == 0)) return;
    
    /* Initial guess for the flat point is the distorted point, assuming 
       distortion is small. */
    rq = sqrt (x*x + y*y);
    xq = x; yq = y;
    
    do {
        r = rq;
        xq = (x + yq*sin(ap.she))/ap.scx
            - xq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
            - ap.p1 * (r*r + 2*xq*xq) - 2*ap.p2*xq*yq;
        yq = y/cos(ap.she)
            - yq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
            - ap.p2 * (r*r + 2*yq*yq) - 2*ap.p1*xq*yq;
        rq = sqrt (xq*xq + yq*yq);
        
        /* Limit divergent iteration. I realize these are "magic values" here
           but they should work in most cases and trying to automatically find
           non-magic values will slow this down considerably. 
        */
        if (rq > 1.2*r) rq = 0.5*r;
        
        itnum++;
    } while ((fabs(rq - r)/r > tol) && (itnum < 201));
    
    /* Final step uses the iteratively-found R and x, y to apply the exact
       correction, equivalent to one more iteration. */
    r = rq;
    *x1 = (x + yq*sin(ap.she))/ap.scx
	    - xq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
    	- ap.p1 * (r*r + 2*xq*xq) - 2*ap.p2*xq*yq;
    *y1 = y/cos(ap.she)
	    - yq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r)
    	- ap.p2 * (r*r + 2*yq*yq) - 2*ap.p1*xq*yq;
}



/*  the following 2 functions may seem trivial, but history proved that
    it's easy to confuse the order of operations here. 
*/


/*  flat_to_dist() Converts 2D metric coordinates from flat-image, 
    centered-sensor representation of ideal camera (representing e.g. ray 
    tracing results from 3D coordinates to camera plane) to real (metric) image
    coordinates in a camera with sensor shift and radial/decentering distortions.
    
    Arguments:
    double flat_x, flat_y - input metric flat-image coordinates.
    Calibration *cal - camera parameters including sensor shift and distortions.
    double *dist_x, *dist_y - output metric real-image coordinates.
*/
void flat_to_dist(double flat_x, double flat_y, Calibration *cal, 
    double *dist_x, double *dist_y)
{
    /* Make coordinates relative to sessor center rather than primary point
       image coordinates, because distortion formula assumes it, [1] p.180 */
    flat_x += cal->int_par.xh;
    flat_y += cal->int_par.yh;
    
    distort_brown_affin(flat_x, flat_y, cal->added_par, dist_x, dist_y);
}

/*  dist_to_flat() attempts to restore metric flat-image positions from metric 
    real-image coordinates. This is an inverse problem so some error is to be
    expected, but for small enough distortions it's bearable.
    
    Arguments:
    double dist_x, dist_y - input metric real-image coordinates.
    Calibration *cal - camera parameters including sensor shift and distortions.
    double *flat_x, *flat_y - output metric flat-image coordinates.
    double tol - tolerance of the radial distance of found point from image 
        center as acceptable percentage of improvement between iterations, 
        under which we can stop.
*/
void dist_to_flat(double dist_x, double dist_y, Calibration *cal,
    double *flat_x, double *flat_y, double tol) 
{
    correct_brown_affine_exact(dist_x, dist_y, cal->added_par, flat_x, flat_y,
        tol);
    *flat_x -= cal->int_par.xh;
    *flat_y -= cal->int_par.yh;
}

