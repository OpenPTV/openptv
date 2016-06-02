#ifndef TRAFO_H
#define TRAFO_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"
#include "lsqadj.h"


/* preliminary remapping of the y pixel coordinate */
typedef enum {
  NO_REMAP,
  DOUBLED_PLUS_ONE,
  DOUBLED
} y_remap_mode_t;


/*  wraps previous one, parameters are read directly from control_par* structure */
void pixel_to_metric(double * x_metric
				 , double * y_metric
				 , double x_pixel
				 , double y_pixel
				 , control_par* parameters				 
				 );


/*  transformation detection geometric coordinates -> pixel coordinates */

void metric_to_pixel(double * x_pixel
				 , double * y_pixel
				 , double x_metric
				 , double y_metric
				 , control_par* parameters				 
				 );

/* correct for Brown affine transformation */
void correct_brown_affin (double x
                        , double y
                        , ap_52 ap
                        , double *x1
                        , double *y1);
     
/* distort according to affine transformation */                        
void distort_brown_affin (double x
                        , double y
                        , ap_52 ap
                        , double *x1
                        , double *y1);

void correct_brown_affine_exact(double x, double y, ap_52 ap, 
    double *x1, double *y1, double tol);

void flat_to_dist(double flat_x, double flat_y, Calibration *cal, 
    double *dist_x, double *dist_y);
void dist_to_flat(double dist_x, double dist_y, Calibration *cal,
    double *flat_x, double *flat_y, double tol);

/* For testing only, please don't use these directly. */
void old_pixel_to_metric (double *x_metric, double *y_metric, double x_pixel,
    double y_pixel,int im_size_x, int im_size_y, double pix_size_x, 
    double pix_size_y, int y_remap_mode);
void old_metric_to_pixel (double * x_pixel, double * y_pixel, double x_metric,
    double y_metric, int im_size_x, int im_size_y, double pix_size_x,
    double pix_size_y, int y_remap_mode);

#endif

