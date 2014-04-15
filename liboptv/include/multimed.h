#ifndef MULTIMED_H
#define MULTIMED_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <optv/calibration.h>
#include <optv/parameters.h>
#include "ray_tracing.h"
#include "trafo.h"
#include <optv/tracking_frame_buf.h>
#include "lsqadj.h"


typedef struct {
    double x, y, z;
} Origin; 

/* mmLUT structure */
typedef struct {
    Origin origin;
    int    nr, nz, rw;
    double *data; 
} mmlut;



double get_mmf_from_mmLUT (int i_cam, double X, double Y, double Z, mmlut *mmLUT);

/* Note that multimed_nlay_v2 is renamted to _nlay) */
void  multimed_nlay (Exterior *ex
                   , mm_np *mm
                   , double X
                   , double Y
                   , double Z
                   , double *Xq
                   , double *Yq
                   , int i_cam
                   , mmlut *mmLUT);

/* Note that multimed_r_nlay_v2 is renamed */
double multimed_r_nlay (Exterior *ex
                      , mm_np *mm
                      , double X
                      , double Y
                      , double Z
                      , int i_cam
                      , mmlut *mmLUT);

void init_mmLUT (volume_par *vpar
               , control_par *cpar
               , Calibration *cal
               , mmlut *mmLUT);

void volumedimension (double *xmax
					, double *xmin
					, double *ymax
					, double *ymin
					, double *zmax
					, double *zmin
					, volume_par *vpar
               		, control_par *cpar
               		, Calibration *cal);

void trans_Cam_Point(Exterior ex, mm_np mm, Glass gl, double X, double Y, double Z, \
Exterior *ex_t, double *X_t, double *Y_t, double *Z_t, double cross_p[3], double cross_c[3]);

void back_trans_Point(double X_t, double Y_t, double Z_t, mm_np mm, Glass G, \
double cross_p[3], double cross_c[3], double *X, double *Y, double *Z);


#endif

