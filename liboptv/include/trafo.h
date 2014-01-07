#ifndef TRAFO_H
#define TRAFO_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"
#include "lsqadj.h"

void pixel_to_metric (xp,yp, imx,imy, pix_x,pix_y, xc,yc, field);
void metric_to_pixel (xc,yc, imx,imy, pix_x,pix_y, xp,yp, field);
void distort_brown_affin (x, y, ap, x1, y1);
void correct_brown_affin (x, y, ap, x1, y1);


#endif

