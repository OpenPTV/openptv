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


double get_mmf_from_mmlut (Calibration *cal, vec3d pos);

void  multimed_nlay (Calibration *cal, mm_np *mm, vec3d pos, double *Xq, 
    double *Yq);

double multimed_r_nlay (Calibration *cal, mm_np *mm, vec3d pos);

void init_mmlut (volume_par *vpar, control_par *cpar, Calibration *cal);

void volumedimension (double *xmax
					, double *xmin
					, double *ymax
					, double *ymin
					, double *zmax
					, double *zmin
					, volume_par *vpar
               		, control_par *cpar
               		, Calibration **cal);

void trans_Cam_Point(Exterior ex, mm_np mm, Glass gl, vec3d pos, Exterior *ex_t, 
    vec3d pos_t, double cross_p[3], double cross_c[3]);

void back_trans_Point(vec3d pos_t, mm_np mm, Glass G, double cross_p[3], 
    double cross_c[3], vec3d pos);

void move_along_ray(double glob_Z, vec3d vertex, vec3d direct, vec3d out);

#endif

