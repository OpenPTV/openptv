/* Utilities for handling sortgrid */

#ifndef SORTGRID_H
#define SORTGRID_H

#include "calibration.h"
#include "parameters.h"
#include "tracking_frame_buf.h"
#include "imgcoord.h"
#include "multimed.h"   
#include "trafo.h"
#include "orientation.h"
#include "vec_utils.h"
#include <stdio.h>

			   
target* sortgrid (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], int num,
    int eps, target pix[]);				
int nearest_neighbour_pix (target pix[], int num, double x, double y, double eps);
int read_sortgrid_par(char *filename);
vec3d *read_calblock(int* num_points, char* filename);

#endif

