#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "tracking_frame_buf.h"

typedef struct {
    /* ... structure definition ... */
} control_par;

typedef struct {
    /* ... structure definition ... */
} volume_par;

typedef struct {
    /* ... structure definition ... */
} track_par;

/* Function declarations */
control_par* new_control_par(int num_cams);
control_par* read_control_par(char* filename);
void c_free_control_par(control_par *cp);  /* Note: it's actually c_free_control_par */
int compare_control_par(control_par *cp1, control_par *cp2);

volume_par* new_volume_par(void);
volume_par* read_volume_par(char* filename);
void free_volume_par(volume_par *vp);
int compare_volume_par(volume_par *vp1, volume_par *vp2);

track_par* read_track_par(char* filename);
void free_track_par(track_par *tp);
int compare_track_par(track_par *tp1, track_par *tp2);

#endif