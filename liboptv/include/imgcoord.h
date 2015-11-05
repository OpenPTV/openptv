#ifndef IMGCOORD_H
#define IMGCOORD_H

#include "parameters.h"
#include "calibration.h"
#include "vec_utils.h"

void img_coord (vec3d pos, Calibration *cal, mm_np *mm, double *x, double *y);
void flat_image_coord (vec3d pos, Calibration *cal, mm_np *mm, double *x, double *y);

#endif

