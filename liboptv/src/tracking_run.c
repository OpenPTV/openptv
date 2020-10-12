/* Implement tracking-run class from tracking_run.c */

#include <stdlib.h>
#include "tracking_run.h"
#include "multimed.h"

/* tr_new_legacy() reads tracking-relevant parameters from .par files,
   initializes the frame buffer using the old 3D-PTV naming rules.
   
   Arguments:
   tracking_run *tr - points to the TrackingRun object to initialize.
   char *seq_par_fname - path to sequence parameters file.
   char *tpar_fname - path to tracking parameters file.
   char *vpar_fname - path to volume parameters file.
   char *cpar_fname - path to control parameters file.
*/
tracking_run* tr_new_legacy(char *seq_par_fname, char *tpar_fname,
    char *vpar_fname, char *cpar_fname, Calibration **cal) 
{
    control_par *cpar = read_control_par(cpar_fname);
    sequence_par *seq_par = read_sequence_par(seq_par_fname, cpar->num_cams);
    return tr_new(seq_par, read_track_par(tpar_fname), 
        read_volume_par(vpar_fname), cpar, 4, 20000,
        "res/rt_is", "res/ptv_is", "res/added", cal, 10000);
}

/* tr_new() aggregates several parameter structs used by tracking, and
   allocates necessary strucures like the frame buffer. It then initializes
   some tracking related metadata. 
   
   There are a lot of arguments to this one, partly because of the frame
   buffer initialization. In the future, when the frame buffer details are
   abstracted away, we can just pass one in. But this is way in the future.
   
   Arguments:
   tracking_run *tr - points to the TrackingRun object to initialize.
   sequence_par *seq_par - sequence parameters.
   track_par *tpar - tracking parameters.
   volume_par *vpar - volume parameters.
   control_par *cpar - control parameters, such as sensor size etc.
   int buf_len - how many consecutive frames to hold in the buffer.
   int max_targets - number of targets to make place for in each buffer.
   char *corres_file_base, *linkage_file_base, *prio_file_base
      - naming scheme in the frame buffer, passed forward
      without tampering. See tracking_frame_buf.c:fb_init()
   Calibration **cal - camra positions etc.
   double flatten_tol - tolerance for the action of transforming distorted 
      image coordinates to flat coordinates.
*/
tracking_run *tr_new(sequence_par *seq_par, track_par *tpar,
    volume_par *vpar, control_par *cpar, int buf_len, int max_targets,
    char *corres_file_base, char *linkage_file_base, char *prio_file_base, 
    Calibration **cal, double flatten_tol)
{
    tracking_run *tr = (tracking_run *) malloc(sizeof(tracking_run));
    tr->tpar = tpar;
    tr->vpar = vpar;
    tr->cpar = cpar;
    tr->seq_par = seq_par;
    tr->cal = cal;
    tr->flatten_tol = flatten_tol;
    
    tr->fb = (framebuf_base*) malloc(sizeof(framebuf));
    fb_init((framebuf*)tr->fb, buf_len, cpar->num_cams, max_targets,
        corres_file_base, linkage_file_base, prio_file_base, seq_par->img_base_name);
    
    tr->lmax = norm((tpar->dvxmin - tpar->dvxmax), \
                    (tpar->dvymin - tpar->dvymax), \
                    (tpar->dvzmin - tpar->dvzmax));
    volumedimension(&(vpar->X_lay[1]), &(vpar->X_lay[0]), &(tr->ymax),
                    &(tr->ymin), &(vpar->Zmax_lay[1]), &(vpar->Zmin_lay[0]),
                    vpar, cpar, cal);
    
    tr->npart = 0;
    tr->nlinks = 0;
    
    return tr;
}

/* tr_free deallocates all data allocated inside a TrackingRun object (but NOT
   the TrackingRun object itself).
   Arguments:
   TrackingRun *tr - points to the TrackingRun object to free.
*/
void tr_free(tracking_run *tr) {
    free(tr->fb);
    free(tr->seq_par->img_base_name);
    free(tr->seq_par);
    free(tr->tpar);
    free(tr->vpar);
    free_control_par(tr->cpar);
}
