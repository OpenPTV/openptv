/* Utilities for handling sortgrid */

#ifndef SORTGRID_H
#define SORTGRID_H

#include "calibration.h"
#include "parameters.h"
#include "tracking_frame_buf.h"
#include "imgcoord.h"
#include "multimed.h"   
#include "trafo.h"
#include "vec_utils.h"
#include <stdio.h>

// to be removed when #78 is merged
typedef double vec2d[2];

				   
void sortgrid (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], int num,
    int eps, target pix[]);
    				
void nearest_pixel_location (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], 
vec2d calib_points[]);
				
int nearest_neighbour_pix (target pix[], int num, double x, double y, double eps);
int read_sortgrid_par(char *filename);
int read_calblock(vec3d fix[], char* filename);

#endif

