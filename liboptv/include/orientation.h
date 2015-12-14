/* Function declarations related to calibration finding routines. */

#ifndef ORENTATION_H
#define ORENTATION_H

#include "vec_utils.h"
#include "calibration.h"
#include "parameters.h"

typedef double vec2d[2];

/* This one is exported for testing purposes. It is only used by the 
   orientation code itself. */
double ray_distance(vec3d vert1, vec3d direct1, vec3d vert2, vec3d direct2);
double epipolar_convergence(vec2d* targets[], int num_targs, int num_cams,
    mm_np *multimed_pars, Calibration* cals[]);

#endif
