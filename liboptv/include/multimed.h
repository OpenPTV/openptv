#ifndef MULTIMED_H
#define MULTIMED_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "calibration.h"
#include "parameters.h"
#include "ray_tracing.h"
#include "trafo.h"
#include "tracking_frame_buf.h"
#include "lsqadj.h"
#include "vec_utils.h"


typedef struct {
    double x, y, z;
} Origin; 

/* mmLUT structure */
typedef struct {
    Origin origin;
    int    nr, nz, rw;
    double *data; 
} mmlut;



double get_mmf_from_mmLUT (mmlut *mmLUT, int i_cam, vec3d pos);

/* Note that multimed_nlay_v2 is renamted to _nlay) */
void  multimed_nlay (mmlut *mmLUT, Exterior *ex, mm_np *mm, vec3d pos, double *Xq, 
    double *Yq, int i_cam);

/* Note that multimed_r_nlay_v2 is renamed */
double multimed_r_nlay (mmlut *mmLUT, Exterior *ex, mm_np *mm, vec3d pos, int i_cam);

void init_mmLUT (mmlut *mmLUT, volume_par *vpar, control_par *cpar, Calibration *cal);

void volumedimension (double *xmax
					, double *xmin
					, double *ymax
					, double *ymin
					, double *zmax
					, double *zmin
					, volume_par *vpar
               		, control_par *cpar
               		, Calibration *cal);

void trans_Cam_Point(Exterior ex, mm_np mm, Glass gl, vec3d pos, Exterior *ex_t, 
    vec3d pos_t, double cross_p[3], double cross_c[3]);

void back_trans_Point(vec3d pos_t, mm_np mm, Glass G, double cross_p[3], 
    double cross_c[3], vec3d pos);


#endif

