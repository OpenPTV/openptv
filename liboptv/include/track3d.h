#ifndef TRACK3D_H
#define TRACK3D_H

#include "parameters.h"
#include "vec_utils.h"
#include "imgcoord.h"
#include "multimed.h"
#include "orientation.h"
#include "calibration.h"
#include "track.h"
#include "tracking_frame_buf.h"


void track3d_loop(tracking_run *run_info, int step);
int find_candidates_in_3d(frame *frm, vec3d pos, double dx, double dy, double dz, int *indices, int max_cands);


#endif // TRACK3D_H