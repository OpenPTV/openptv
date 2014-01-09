#ifndef MULTIMED_H
#define MULTIMED_H

#include <math.h>
#include "parameters.h"
#include "calibration.h"
#include <optv/tracking_frame_buf.h>
#include "ray_tracing.h"
#include "lsqadj.h"

double get_mmf_from_mmLUT (int i_cam, double X, double Y, double Z);

/* Note that multimed_nlay_v2 is renamted to _nlay) */
void  multimed_nlay (Exterior ex, Exterior ex_o, mm_np mm, \
double X, double Y, double Z, double *Xq, double *Yq);

void back_trans_Point_back(double X_t, double Y_t, double Z_t,mm_np mm, Glass G, \
double cross_p[], double cross_c[], double *X, double *Y, double *Z);

void trans_Cam_Point_back(Exterior x,mm_np mm,Glass gl, double X, double Y, double Z,\
Exterior *ex_t, double *X_t, double *Y_t, double *Z_t, double *cross_p, double *cross_c);

void trans_Cam_Point(Exterior ex, mm_np mm, Glass gl, double X, double Y, double Z, \
Exterior *ex_t, double *X_t, double *Y_t, double *Z_t, double *cross_p, double *cross_c);

void back_trans_Point(double X_t, double Y_t, double Z_t, mm_np mm, Glass G, \
double cross_p[], double cross_c[], double *X, double *Y, double *Z);

/* Note that multimed_r_nlay_v2 is renamed */
double multimed_r_nlay (Exterior ex, Exterior ex_o, mm_np mm, double X, double Y,\
double Z);

void init_mmLUT (int i_cam);

void volumedimension (double *xmax, double *xmin, double *ymax, double *ymin, \
double *zmax, double *zmin, int num_cams);




#endif

