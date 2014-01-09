/*  Objects related to calibration: exterior and interior parameters, glass
    parameters. A Calibration object to rule them all and find them. 

    References:
    [1] Th. Dracos (editor). Three-Dimensional Velocity and Vorticity Measuring
        and Image Analysis Techniques. Kluwer Academic Publishers. 1996. 
        pp.165-168
*/

#ifndef RAYTRACING_H
#define RAYTRACING_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"
#include "lsqadj.h"

void ray_tracing (double x
				, double y
				, Calibration* cal
				, mm_np mm
				, double X[3]
				, double a[3]);
				
void old_ray_tracing (double x, double y, Exterior Ex, Interior I, Glass G,\
                  mm_np mm, double *Xb2, double *Yb2, double *Zb2, \
                  double *a3, double *b3,double *c3);


#endif

