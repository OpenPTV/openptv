/*******************************************************************************
**
** Title: ptv
**
** Author: Heinrich Stueer
**
** Description: Main modul of track.
** Dies ist eine abgespeckte Version vom Malik und Papantoniou (allerdings mit
** ein paar Anderungen)
** Created: 12.02.1998
** Changes:
**
*******************************************************************************/
/*
Copyright (c) 1990-2011 ETH Zurich

See the file license.txt for copying permission.
*/

/* ----- recent changes -------------------------------------------------------
  ad holten, 12-2012
  replaced:
        if		(filenumber < 10)	sprintf (filein,
"res/rt_is.%1d", filenumber); else if (filenumber < 100)	sprintf (filein,
"res/rt_is.%2d",  filenumber);
        else						sprintf (filein,
"res/rt_is.%3d", filenumber); by sprintf (filein, "res/rt_is.%d", filenumber);

  replaced:
        fp = fopen (filein, "r");
        if (! fp) printf("Can't open ascii file: %s\n", filein);
  by
        fp = fopen_rp (filein);		(fopen_rp prints an error message on
failure) if (!fp) return;
------------------------------------------------------------------------------*/

#include "tracking_run.h"
#include "track3d.h"
#include "track.h"
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

// track3d_loop is the main tracking subroutine for 3D candidate search, designed to be interchangeable with trackcorr_c_loop
void track3d_loop(tracking_run *run_info, int step) {
    // Shortcuts into the tracking_run struct
    framebuf_base *fb = run_info->fb;
    track_par *tpar = run_info->tpar;

    frame *prev = fb->buf[-1];
    frame *curr = fb->buf[0];
    frame *next = fb->buf[1];

    // frame *prev = fb_get_frame(fb, 0);
    // frame *curr = fb_get_frame(fb, 1);
    // frame *next = fb_get_frame(fb, 2);
    int i, d, k;
    int orig_parts = curr->num_parts;
    int cand_indices[MAX_CANDS];
    float decis[MAX_CANDS];
    int linkdecis[MAX_CANDS];
    vec3d predicted, vel;
    int nvel;
    
    double dx = tpar->dvxmax;
    double dy = tpar->dvymax;
    double dz = tpar->dvzmax;

    // Level 1: Particles with previous links
    for (i = 0; i < orig_parts; i++) {
        P *curr_path_inf = &curr->path_info[i];
        if (curr_path_inf->prev < 0) continue;
        int prev_idx = curr_path_inf->prev;
        if (prev_idx < 0 || prev_idx >= prev->num_parts) continue;
        P *prev_path_inf = &prev->path_info[prev_idx];
        for (d = 0; d < 3; d++)
            predicted[d] = 2 * curr_path_inf->x[d] - prev_path_inf->x[d];
        int num_cands = find_candidates_in_3d(next, predicted, dx, dy, dz, cand_indices, MAX_CANDS);
        for (k = 0; k < num_cands; k++) {
            float acc = 0.0f;
            for (d = 0; d < 3; d++) {
                float diff = curr_path_inf->x[d] - 2 * next->path_info[cand_indices[k]].x[d] + prev_path_inf->x[d];
                acc += diff * diff;
            }
            decis[k] = sqrtf(acc);
            linkdecis[k] = cand_indices[k];
        }
        if (num_cands > 1) {
            sort(num_cands, decis, linkdecis);
        }
        if (num_cands > 0 && next->path_info[linkdecis[0]].prev < 0) {
            curr_path_inf->next = linkdecis[0];
            next->path_info[linkdecis[0]].prev = i;
        } else {
            curr_path_inf->next = -1;
        }
    }

    // Level 2: No previous link, but neighbors have previous links
    for (i = 0; i < orig_parts; i++) {
        P *curr_path_inf = &curr->path_info[i];
        if (curr_path_inf->prev >= 0 || curr_path_inf->next >= 0) continue;
        nvel = 0;
        for (d = 0; d < 3; d++) vel[d] = 0.0;
        for (int j = 0; j < orig_parts; j++) {
            if (j == i) continue;
            P *nbr = &curr->path_info[j];
            if (fabs(nbr->x[0] - curr_path_inf->x[0]) < dx &&
                fabs(nbr->x[1] - curr_path_inf->x[1]) < dy &&
                fabs(nbr->x[2] - curr_path_inf->x[2]) < dz &&
                nbr->prev >= 0) {
                for (d = 0; d < 3; d++)
                    vel[d] += nbr->x[d] - prev->path_info[nbr->prev].x[d];
                nvel++;
            }
        }
        if (nvel == 0) continue;
        for (d = 0; d < 3; d++) vel[d] /= nvel;
        for (d = 0; d < 3; d++)
            predicted[d] = curr_path_inf->x[d] + vel[d];
        int num_cands = find_candidates_in_3d(next, predicted, dx, dy, dz, cand_indices, MAX_CANDS);
        for (k = 0; k < num_cands; k++) {
            float acc = 0.0f;
            for (d = 0; d < 3; d++) {
                float diff = curr_path_inf->x[d] - 2 * next->path_info[cand_indices[k]].x[d] + predicted[d];
                acc += diff * diff;
            }
            decis[k] = sqrtf(acc);
            linkdecis[k] = cand_indices[k];
        }
        if (num_cands > 1) {
            sort(num_cands, decis, linkdecis);
        }
        if (num_cands > 0 && next->path_info[linkdecis[0]].prev < 0) {
            curr_path_inf->next = linkdecis[0];
            next->path_info[linkdecis[0]].prev = i;
        } else {
            curr_path_inf->next = -1;
        }
    }

    // Level 3: No previous link, no neighbors with previous links
    for (i = 0; i < orig_parts; i++) {
        P *curr_path_inf = &curr->path_info[i];
        if (curr_path_inf->prev >= 0 || curr_path_inf->next >= 0) continue;
        for (d = 0; d < 3; d++)
            predicted[d] = curr_path_inf->x[d];
        int num_cands = find_candidates_in_3d(next, predicted, dx, dy, dz, cand_indices, MAX_CANDS);
        for (k = 0; k < num_cands; k++) {
            float acc = 0.0f;
            for (d = 0; d < 3; d++) {
                float diff = curr_path_inf->x[d] - 2 * next->path_info[cand_indices[k]].x[d] + predicted[d];
                acc += diff * diff;
            }
            decis[k] = sqrtf(acc);
            linkdecis[k] = cand_indices[k];
        }
        if (num_cands > 1) {
            sort(num_cands, decis, linkdecis);
        }
        if (num_cands > 0 && next->path_info[linkdecis[0]].prev < 0) {
            curr_path_inf->next = linkdecis[0];
            next->path_info[linkdecis[0]].prev = i;
        } else {
            curr_path_inf->next = -1;
        }
    }
}

// Returns the number of candidates found within a 3D box centered at pos
int find_candidates_in_3d(frame *frm, vec3d pos, double dx, double dy, double dz, int *indices, int max_cands) {
    int i, count = 0;
    for (i = 0; i < frm->num_parts; i++) {
        if (fabs(frm->path_info[i].x[0] - pos[0]) < dx &&
            fabs(frm->path_info[i].x[1] - pos[1]) < dy &&
            fabs(frm->path_info[i].x[2] - pos[2]) < dz) {
            if (count < max_cands) {
                indices[count++] = i;
            }
        }
    }
    return count;
}
