/*  Objects related to calibration: exterior and interior parameters, glass
    parameters. A Calibration object to rule them all and find them. 

    References:
    [1] Th. Dracos (editor). Three-Dimensional Velocity and Vorticity Measuring
        and Image Analysis Techniques. Kluwer Academic Publishers. 1996. 
        pp.165-168
*/

#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"
#include "lsqadj.h"
#include "math.h"

/* TODO: replace local functions by vec_utils */

void modu(double a[3], double *m);
void norm_cross(double a[3], double b[3], double *n1, double *n2, double *n3);
void dot(double a[3], double b[3], double *d);



void ray_tracing_v2 (double x, double y,Exterior Ex, Interior I, Glass G, mm_np mm,\
double *Xb2, double *Yb2, double *Zb2, double *a3, double *b3, double *c3); 




#endif

