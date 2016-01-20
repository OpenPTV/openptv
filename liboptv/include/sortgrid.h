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

			       
void sortgrid (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], int num,
    int eps, target pix[]);
    				
void pixel_location (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], 
vec2d calib_points[]);
				
int read_sortgrid_par(char *filename);
int read_calblock(vec3d fix[], char* filename);

double dist (vec2d calib_point, target pix, double eps);
int *sorted_order (const double *arr, int n);
void swap(int *a, int *b);

#endif

