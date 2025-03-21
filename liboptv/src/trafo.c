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
void distort_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1) {
    double r = sqrt(x*x + y*y);
    
    if (r < 1e-10) {
        *x1 = 0;
        *y1 = 0;
        return;
    }
    
    // Apply radial distortion
    double r2 = r*r;
    double r4 = r2*r2;
    double r6 = r4*r2;
    double radial_factor = 1.0 + ap.k1*r2 + ap.k2*r4 + ap.k3*r6;
    
    // Apply decentering distortion
    double x_dist = x * radial_factor + ap.p1 * (r2 + 2*x*x) + 2*ap.p2*x*y;
    double y_dist = y * radial_factor + ap.p2 * (r2 + 2*y*y) + 2*ap.p1*x*y;
    
    // Apply affine transformation more stably
    double sin_she = sin(ap.she);
    double cos_she = cos(ap.she);
    
    // Scale first, then shear to maintain better numerical precision
    *x1 = ap.scx * (x_dist - sin_she * y_dist);
    *y1 = ap.scx * cos_she * y_dist;  // Apply same scale to maintain aspect ratio
}

/*  correct distorted to flat-image coordinates, see correct_brown_affine_exact(),
    this one is the same except it ensures only one iteration for backward 
    compatibility.
*/
void correct_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1) {
    // Use a more stable iteration scheme with better initial guess
    const double sin_she = sin(ap.she);
    const double cos_she = cos(ap.she);
    const double inv_scx = 1.0/ap.scx;
    
    // Initial guess: inverse affine transformation
    double xq = x * inv_scx;
    double yq = y * inv_scx / cos_she;  // Corrected for scale
    xq += yq * sin_she;  // Apply shear correction
    
    const int MAX_ITER = 20;
    const double DAMPING = 0.7;  // Adjusted damping factor
    const double TOL = 1e-8;  // Tight tolerance for convergence
    
    for (int i = 0; i < MAX_ITER; i++) {
        // Store previous values for convergence check
        double xq_old = xq;
        double yq_old = yq;
        
        // Calculate distorted position for current guess
        double xt, yt;
        distort_brown_affin(xq, yq, ap, &xt, &yt);
        
        // Calculate error
        double dx = (x - xt) * inv_scx;
        double dy = (y - yt) * inv_scx;
        
        // Update estimate with damping
        xq += dx * DAMPING;
        yq += dy * DAMPING;
        
        // Check convergence using relative error
        double change = sqrt((xq - xq_old)*(xq - xq_old) + 
                           (yq - yq_old)*(yq - yq_old));
        double pos_magnitude = sqrt(xq*xq + yq*yq);
        if (pos_magnitude > 1e-10 && change/pos_magnitude < TOL) {
            break;
        }
    }
    
    *x1 = xq;
    *y1 = yq;
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
    const double r_init = sqrt(x*x + y*y);
    
    if (r_init < 1e-10) {
        *x1 = 0;
        *y1 = 0;
        return;
    }
    
    // Pre-compute trig values
    const double sin_she = sin(ap.she);
    const double cos_she = cos(ap.she);
    const double inv_scx = 1.0/ap.scx;
    
    // Initial guess: inverse affine transformation
    double xq = (x + y*sin_she)*inv_scx;
    double yq = y/cos_she;
    
    const int MAX_ITER = 50;  // Increased for large shear angles
    const double DAMPING = 0.5;  // More conservative damping
    
    for (int i = 0; i < MAX_ITER; i++) {
        double r2 = xq*xq + yq*yq;
        double r4 = r2*r2;
        double r6 = r4*r2;
        
        // Calculate distortion factors
        double radial_factor = ap.k1*r2 + ap.k2*r4 + ap.k3*r6;
        
        // Calculate correction terms
        double dx = xq * radial_factor + ap.p1 * (r2 + 2*xq*xq) + 2*ap.p2*xq*yq;
        double dy = yq * radial_factor + ap.p2 * (r2 + 2*yq*yq) + 2*ap.p1*xq*yq;
        
        // Calculate new estimates
        double xq_new = (x + y*sin_she)*inv_scx - dx;
        double yq_new = y/cos_she - dy;
        
        // Apply damping
        double dx_change = xq_new - xq;
        double dy_change = yq_new - yq;
        
        xq += DAMPING * dx_change;
        yq += DAMPING * dy_change;
        
        // Check convergence
        if (sqrt(dx_change*dx_change + dy_change*dy_change) < tol) {
            break;
        }
    }
    
    *x1 = xq;
    *y1 = yq;
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

