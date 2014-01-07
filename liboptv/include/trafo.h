#ifndef TRAFO_H
#define TRAFO_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"
#include "lsqadj.h"

void pixel_to_metric (double xp, double yp, int imx, int imy, double pix_x, \
double pix_y, double *xc, double *yc, int field);
void metric_to_pixel (double xc, double yc, int imx, int imy, double pix_x, \
double pix_y, double *xp, double *yp, int field);
void correct_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1);
void distort_brown_affin (double x, double y, ap_52 ap, double *x1, double *y1);


#endif

