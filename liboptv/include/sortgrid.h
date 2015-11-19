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

typedef struct
{
  int pnr;
  vec3d pos;
} coord_3d;


typedef struct
{
  double x, y;
} pixel_pos;

				   
void sortgrid (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], int num,
    int eps, target pix[]);
    				
void nearest_pixel_location (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], 
pixel_pos calib_points[]);
				
int nearest_neighbour_pix (target pix[], int num, double x, double y, double eps);
int read_sortgrid_par(char *filename);
int read_calblock(coord_3d fix[], char* filename);

#endif

