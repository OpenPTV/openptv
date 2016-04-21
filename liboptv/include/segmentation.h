/*  Objects related to calibration: exterior and interior parameters, glass
    parameters. A Calibration object to rule them all and find them. 

    References:
    [1] Th. Dracos (editor). Three-Dimensional Velocity and Vorticity Measuring
        and Image Analysis Techniques. Kluwer Academic Publishers. 1996. 
        pp.165-168
*/

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "tracking_frame_buf.h"
#include "parameters.h" 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


typedef struct
{
  int 	pos, status;
  short	xmin, xmax, ymin, ymax;
  int   n, sumg;
  double  x, y;
  int   unr, touch[4], n_touch;	/* unified with target unr, touching ... */
}
peak;

typedef struct
{
  short	       	x,y;
  unsigned char	g;
  short	       	tnr;
}
targpix;


int peak_fit_new ( unsigned char *img, target_par *trgtpar, int xmin, int xmax, int ymin, 
int ymax, target pix[], int nr, control_par *cpar); 

void check_touch (peak *tpeak, int p1, int p2);

void simple_connectivity(unsigned char *img, char par_file[],
    int xmin, int xmax, int ymin, int ymax,
    target pix[], int nr, int *num, control_par *cpar);

void targ_rec (unsigned char *img, char par_file[], 
    int xmin, int xmax, int ymin, int ymax,
    target pix[], int nr, int *num, control_par *cpar);

#endif

