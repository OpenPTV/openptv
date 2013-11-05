#ifndef TRAFO_H
#define TRAFO_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include <optv/tracking_frame_buf.h>
#include "lsqadj.h"


/* preliminary remapping of the y pixel coordinate */
typedef enum {
  NO_REMAP,
  DOUBLED_PLUS_ONE,
  DOUBLED
} y_remap_mode_t;


/*  transformation detection pixel coordinates -> geometric coordinates */
void pixel_to_metric (double * x_metric
		       , double * y_metric
		       , double x_pixel
		       , double y_pixel
		       , int im_size_x
		       , int im_size_y
		       , double pix_size_x
		       , double pix_size_y
		       , int y_remap_mode);

/*  wraps previous one, parameters are read directly from control_par* structure */
void pixel_to_metric_control_par(double * x_metric
				 , double * y_metric
				 , double x_pixel
				 , double y_pixel
				 , control_par* parameters				 
				 );


/*  transformation detection geometric coordinates -> pixel coordinates */
void metric_to_pixel (double * x_pixel
		      , double * y_pixel
		      , double x_metric
		      , double y_metric
		      , int im_size_x
		      , int im_size_y
		      , double pix_size_x
		      , double pix_size_y
		      , int y_remap_mode);

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


#endif

