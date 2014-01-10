#ifndef MULTIMED_H
#define MULTIMED_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "optv/calibration.h"
#include <optv/parameters.h>
#include "ray_tracing.h"
#include "trafo.h"
#include <optv/tracking_frame_buf.h>
#include "lsqadj.h"


/* mmLUT structure */
typedef struct {
    int num_cams;
    char **img_base_name; /* Note the duplication with sequence_par. */
    char **cal_img_base_name;
    int hp_flag;
    int allCam_flag;
    int tiff_flag;
    int imx;
    int imy;
    double pix_x;
    double pix_y;
    int chfield; 
    mm_np *mm; 
} mmlut;



double get_mmf_from_mmLUT (int i_cam, double X, double Y, double Z);

/* Note that multimed_nlay_v2 is renamted to _nlay) */
void  multimed_nlay (Exterior ex, Exterior ex_o, mm_np mm, \
double X, double Y, double Z, double *Xq, double *Yq, int cam);

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
double Z, int cam);

void init_mmLUT (int i_cam, volume_par *vpar, control_par *cpar, mmlut mmLUT);

void volumedimension (double *xmax, double *xmin, double *ymax, double *ymin, \
double *zmax, double *zmin, int num_cams);




#endif
