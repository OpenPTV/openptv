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


typedef struct
{
  int 	pos, status;
  short	xmin, xmax, ymin, ymax;
  int   n, sumg;
  double  x, y;
  int   unr, touch[4], n_touch;	/* unified with target unr, touching ... */
}
peak;



int peak_fit(unsigned char *img, target_par *targ_par, 
    int xmin, int xmax, int ymin, int ymax, control_par *cpar, int num_cam, 
    target pix[]); 

void check_touch (peak *tpeak, int p1, int p2);

int targ_rec (unsigned char *img, target_par *targ_par, int xmin, 
int xmax, int ymin, int ymax, control_par *cpar, int num_cam, target pix[]);
    

#endif

