/* Utilities for handling sortgrid */

#ifndef SORTGRID_H
#define SORTGRID_H

#include "calibration.h"
#include "parameters.h"
#include "tracking_frame_buf.h"
#include "imgcoord.h"
#include "multimed.h"   
#include "trafo.h"
#include <stdio.h>

typedef struct
{
  int pnr;
  vec3d pos;
} coord_3d;

typedef struct
{
  int x, y;
} pixel_pos;

				   
void sortgrid_man (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], int num,
    target pix[]);
    				
void just_plot (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], 
    pixel_pos calib_points[]);
				
int nearest_neighbour_pix (target pix[], int num, double x, double y, double eps);
int read_sortgrid_par(char *filename);

#endif

