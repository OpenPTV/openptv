/* Definitions for tracking routines. */

#ifndef TRACK_H
#define TRACK_H

#include "tracking_frame_buf.h"
#include "parameters.h"
#include "trafo.h"
#include "tracking_run.h"
#include "vec_utils.h"
#include "imgcoord.h"
#include "multimed.h"
#include "orientation.h"
#include "calibration.h"

/* The buffer space required for this algorithm:
 
 Note that MAX_TARGETS is taken from the global M, but I want a separate
 definition because the fb created here should be local, not used outside
 this file.
 
 MAX_CANDS is the max number of candidates sought in search volume for next
 link.
 */
#define TR_BUFSPACE 4
#define TR_MAX_CAMS 4
#define MAX_TARGETS 20000
#define MAX_CANDS 4         // max candidates, nearest neighbours
#define ADD_PART 3          // search region 3 pix around a particle

typedef struct /* struct for what was found to corres */
{
 int ftnr, freq, whichcam[TR_MAX_CAMS];
}
foundpix;

int candsearch_in_pix(target  next[], int num_targets, double x, double y,
    double dl, double dr, double du, double dd, int p[4], control_par *cpar);
int candsearch_in_pix_rest(target  next[], int num_targets, double x, double y,
    double dl, double dr, double du, double dd, int p[], control_par *cpar);
int sort_candidates_by_freq (foundpix item[16], int num_cams);
void searchquader(vec3d point, double xr[4], double xl[4], double yd[4], \
    double yu[4], track_par *tpar, control_par *cpar, Calibration **cal);
void predict(vec2d a, vec2d b, vec2d c);
void search_volume_center_moving(vec3d prev_pos, vec3d curr_pos, vec3d output);
int pos3d_in_bounds(vec3d pos, track_par *bounds);
void det_lsq_3d (Calibration *cals, mm_np mm, vec2d v[], double *Xp, \
    double *Yp, double *Zp, int num_cams);
void sort(int n, float a[], int b[]);
void angle_acc(vec3d start, vec3d pred, vec3d cand, double *angle, double *acc);
void reset_foundpix_array(foundpix *arr, int arr_len, int num_cams);
void copy_foundpix_array(foundpix *dest, foundpix *src, int arr_len, \
    int num_cams);
void point_to_pixel (vec2d v1, vec3d point, Calibration *cal, control_par *cpar);

void track_forward_start(tracking_run *tr);
void trackcorr_c_loop (tracking_run *run_info, int step);
void trackcorr_c_finish(tracking_run *run_info, int step);
double trackback_c(tracking_run *run_info);

/* add_particle() inserts a particle at a given position to the end of the 
 * frame, along with associated targets.
 * 
 * Arguments:
 * frame *frm - the frame to store the particle.
 * vec3d pos - position of inserted particle in the global coordinates.
 * int cand_inds[][MAX_CANDS] - indices of candidate targets for association
 *    with this particle.
 */
void add_particle(frame *frm, vec3d pos, int cand_inds[][MAX_CANDS]);

#endif
