/*  Objects related to calibration: exterior and interior parameters, glass
    parameters. A Calibration object to rule them all and find them. 

    References:
    [1] Th. Dracos (editor). Three-Dimensional Velocity and Vorticity Measuring
        and Image Analysis Techniques. Kluwer Academic Publishers. 1996. 
        pp.165-168
*/

#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "calibration.h"
#include "tracking_frame_buf.h"

void modu(double a[3], double *m);

void norm_cross(double a[3], double b[3], double *n1, double *n2, double *n3);

void point_line_line(Calibration *c0, double gX0, double gY0, double gZ0, \
double a0, double b0, double c0,\
Calibration *c1, double gX1, double gY1, double gZ1, double a1, \
double b1, double c1, double *x, double *y,double *z);

void ray_tracing (double x,double y,Calibraton *c, double *Xb2,double *Yb2,
double *Zb2, double *a3, double *b3, double *c3);






#endif

