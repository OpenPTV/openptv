/* An object representing a tracking run. Holds parameters but also 
   intermediate results collected on each step of tracking. For use in
   tracking algorithms that need to communicate with sub-components such as
   single-step tracking routine driven by a loop.
*/

#ifndef TRACKING_RUN_H
#define TRACKING_RUN_H

#include "parameters.h"
#include "calibration.h"
#include "tracking_frame_buf.h"

typedef struct {
    framebuf_base *fb;
    sequence_par *seq_par;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;
    Calibration **cal;
    double flatten_tol; /* Tolerance for dist_to_flat() */
    
    /* Intermediate calculations done in run setup phase and used in the loop: */
    double ymin, ymax, lmax;
    int npart, nlinks;
} tracking_run;

tracking_run* tr_new_legacy(char *seq_par_fname, char *tpar_fname,
    char *vpar_fname, char *cpar_fnamei, Calibration **cal);
tracking_run* tr_new(sequence_par *seq_par, track_par *tpar,
    volume_par *vpar, control_par *cpar, int buf_len, int max_targets,
    char *corres_file_base, char *linkage_file_base, char *prio_file_base, 
    Calibration **cal, double flatten_tol);
void tr_free(tracking_run *tr);

#endif
