/* Implement tracking-run class from tracking_run.c */

#include <stdlib.h>
#include "tracking_run.h"

/* tr_init() reads tracking-relevant parameters and allocates space for
   a tracking framebuffer (for now you have to fb_init yourself).
   
   Arguments:
   TrackingRun *tr - points to the TrackingRun object to initialize.
   char *seq_par_fname - path to sequence parameters file.
   char *tpar_fname - path to tracking parameters file.
   char *vpar_fname - path to volume parameters file.
   char *cpar_fname - path to control parameters file.
*/
void tr_init(tracking_run *tr, char *seq_par_fname, char *tpar_fname,
    char *vpar_fname, char *cpar_fname) 
{
    tr->fb = (framebuf *) malloc(sizeof(framebuf));
    tr->tpar = read_track_par(tpar_fname);
    tr->vpar = read_volume_par(vpar_fname);
    tr->cpar = read_control_par(cpar_fname);
    tr->seq_par = read_sequence_par(seq_par_fname, tr->cpar->num_cams);
    tr->npart = 0;
    tr->nlinks = 0;
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
