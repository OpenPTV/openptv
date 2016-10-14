/* An object representing a tracking run. Holds parameters but also 
   intermediate results collected on each step of tracking. For use in
   tracking algorithms that need to communicate with sub-components such as
   single-step tracking routine driven by a loop.
*/

#ifndef TRACKING_RUN_H
#define TRACKING_RUN_H

#include "parameters.h"
#include "tracking_frame_buf.h"

typedef struct {
    framebuf *fb;
    sequence_par *seq_par;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;
    
    /* Intermediate calculations done in run setup phase and used in the loop: */
    double ymin, ymax, lmax;
    int npart, nlinks;
} tracking_run;

void tr_init(tracking_run *tr, char *seq_par_fname, char *tpar_fname,
    char *vpar_fname, char *cpar_fname);
void tr_free(tracking_run *tr);

#endif
