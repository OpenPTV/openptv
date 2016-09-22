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


n_tupel *correspondences (frame *frm, coord_2d **corrected, 
    volume_par *vpar, control_par *cpar, Calibration **calib,
    int match_counts[]);


/* subcomponents of correspondences, may be separately useful. */
int** safely_allocate_target_usage_marks(int num_cams);
void deallocate_target_usage_marks(int** tusage, int num_cams);

int safely_allocate_adjacency_lists(correspond* lists[4][4], int num_cams, 
    int *target_counts);
void deallocate_adjacency_lists(correspond* lists[4][4], int num_cams);

int four_camera_matching(correspond *list[4][4], int base_target_count, 
    double accept_corr, n_tupel *scratch, int scratch_size);
int three_camera_matching(correspond *list[4][4], int num_cams, 
    int *target_counts, double accept_corr, n_tupel *scratch, int scratch_size,
    int** tusage);
int consistent_pair_matching(correspond *list[4][4], int num_cams, 
    int *target_counts, double accept_corr, n_tupel *scratch, int scratch_size,
    int** tusage);

void match_pairs(correspond *list[4][4], coord_2d **corrected, 
    frame *frm, volume_par *vpar, control_par *cpar, Calibration **calib);

#endif
