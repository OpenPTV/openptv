/* Header for the old correspondences code. */

#ifndef CORRESPONDENCES_H
#define CORRESPONDENCES_H

#include "tracking_frame_buf.h"
#include "parameters.h"
#include "calibration.h"

void quicksort_target_y (target *pix, int num);
void quicksort_coord2d_x (coord_2d *crd, int num);
int correspondences_4 (target pix[][nmax], coord_2d geo[][nmax], int num[],
    volume_par *vpar, control_par *cpar, Calibration cals[], n_tupel *con, int match_counts[]);

#endif
