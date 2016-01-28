/* Function declarations related to calibration finding routines. */

#ifndef ORENTATION_H
#define ORENTATION_H

#include "vec_utils.h"
#include "calibration.h"
#include "parameters.h"
#include "tracking_frame_buf.h"
#include "imgcoord.h"
#include "trafo.h"

/* Parameters for orientation */
typedef struct {
    int useflag;
    int ccflag;
    int xhflag;
    int yhflag;
    int k1flag;
    int k2flag;
    int k3flag;
    int p1flag;
    int p2flag;
    int scxflag;
    int sheflag;
    int interfflag;
} orient_par;

typedef double vec2d[2];

/* This one is exported for testing purposes. It is only used by the
   orientation code itself. */
double skew_midpoint(vec3d vert1, vec3d direct1, vec3d vert2, vec3d direct2,
    vec3d res);
double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
    Calibration* cals[], vec3d res);
double weighted_dumbbell_precision(vec2d** targets, int num_targs, int num_cams,
    mm_np *multimed_pars, Calibration* cals[], int db_length, double db_weight);

void orient(Calibration* cal, control_par *cpar, int nfix, vec3d fix[], target pix[], 
            orient_par *flags);
void raw_orient(Calibration* cal, control_par *cpar, int nfix, vec3d fix[], target pix[]);

int read_man_ori_fix(vec3d fix4[4], char* calblock_filename, char* man_ori_filename, 
    int cam);

orient_par* read_orient_par(char *filename);

#endif
