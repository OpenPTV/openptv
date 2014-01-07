#ifndef MULTIMED_H
#define MULTIMED_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"
#include "ray_tracing.h"
#include "lsqadj.h"

double get_mmf_from_mmLUT (int i_cam, double X, double Y, double Z);
void  multimed_nlay (Exterior ex, Exterior ex_o, mm_np mm, \
double X, double Y, double Z, double *Xq, double Yq);


#endif

