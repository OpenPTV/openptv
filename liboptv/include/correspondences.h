/* Header for the old correspondences code. */

#ifndef CORRESPONDENCES_H
#define CORRESPONDENCES_H

#include "tracking_frame_buf.h"
#include "parameters.h"
#include "calibration.h"
#include "epi.h"

#define nmax 202400


typedef struct
{
  int     p[4];
  double  corr;
}
n_tupel;


typedef struct
{
  int    	p1;	       	/* point number of master point */
  int    	n;	       	/* # of candidates */
  int    	p2[MAXCAND];	/* point numbers of candidates */
  double	corr[MAXCAND];	/* feature based correlation coefficient */
  double	dist[MAXCAND];	/* distance perpendicular to epipolar line */
}
correspond;	       	/* correspondence candidates */


void quicksort_target_y (target *pix, int num);
void qs_target_y (target *pix, int left, int right);

void quicksort_coord2d_x (coord_2d *crd, int num);
void qs_coord2d_x (coord_2d	*crd, int left, int right);

void quicksort_con (n_tupel	*con, int num);
void qs_con (n_tupel *con, int left, int right);


n_tupel *correspondences (frame *frm, volume_par *vpar, control_par *cpar, 
Calibration **calib, int match_counts[]);

#endif
