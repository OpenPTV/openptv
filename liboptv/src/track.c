/*******************************************************************

   Routine:	        track.c

   Author/Copyright:        Jochen Willneff

   Address:	        Institute of Geodesy and Photogrammetry
                    ETH - Hoenggerberg
                    CH - 8093 Zurich

   Creation Date:		Beginning: February '98
                        End: far away

   Description:             Tracking of particles in image- and objectspace

   Routines contained:      trackcorr_c

   Updated:           Yosef Meller and Alex Liberzon
   Address:           Tel Aviv University
   For:               OpenPTV, http://www.openptv.net
   Modification date: October 2016

*******************************************************************/

/* References:
   [1] http://en.wikipedia.org/wiki/Gradian
 */

#include "tracking_run.h"
#include "track.h"
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>


/* internal-use defines, not needed by the outside world. */
#define TR_UNUSED -1

/* track_forward_start() - initializes the tracking frame buffer with the 
   first frames.
   
   Arguments:
   tracking_run *tr - an object holding the per-run tracking parameters, and
      a frame buffer with 4 positions.
*/
void track_forward_start(tracking_run *tr) {
    int step;
    /* Prime the buffer with first frames */
    for (step = tr->seq_par->first; 
         step < tr->seq_par->first + TR_BUFSPACE - 1; 
         step++) 
    {
        fb_read_frame_at_end(tr->fb, step, 0);
        fb_next(tr->fb);
    }
    fb_prev(tr->fb);
}

/* reset_foundpix_array() sets default values for foundpix objects in an array.
 *
 * Arguments:
 * foundpix *arr - the array to reset
 * int arr_len - array length
 * int num_cams - number of places in the whichcam member of foundpix.
 */
void reset_foundpix_array(foundpix *arr, int arr_len, int num_cams) {
    int i, cam;
    for (i = 0; i < arr_len; i++) {
        arr[i].ftnr = TR_UNUSED;
        arr[i].freq = 0;
        for(cam = 0; cam < num_cams; cam++) {
            arr[i].whichcam[cam] = 0;
        }
    }
}

/* copy_foundpix_array() copies foundpix objects from one array to another.
 *
 * Arguments:
 * foundpix *dest, *src - src is the array to copy, dest receives it.
 * int arr_len - array length
 * int num_cams - number of places in the whichcam member of foundpix.
 */
void copy_foundpix_array(foundpix *dest, foundpix *src, int arr_len,
                         int num_cams)
{
    int i, cam;
    for (i = 0; i < arr_len; i++) {
        dest[i].ftnr = src[i].ftnr;
        dest[i].freq = src[i].freq;
        for (cam = 0; cam < num_cams; cam++) {
            dest[i].whichcam[cam] = src[i].whichcam[cam];
        }
    }
}

/* register_closest_neighbs() finds candidates for continuing a particle's
 * path in the search volume, and registers their data in a foundpix array
 * that is later used by the tracking algorithm.
 * TODO: the search area can be in a better data structure.
 *
 * Arguments:
 * target *targets - the targets list to search.
 * int num_targets - target array length.
 * int cam - the index of the camera we're working on.
 * double cent_x, cent_y - image coordinates of search area, [pixel]
 * double dl, dr, du, dd - respectively the left, right, up, down distance to
 *   the search area borders from its center, [pixel]
 * foundpix *reg - an array of foundpix objects, one for each possible
 *   neighbour. Output array.
 */
void register_closest_neighbs(target *targets, int num_targets, int cam,
                              double cent_x, double cent_y, double dl, double dr, double du, double dd,
                              foundpix *reg, control_par *cpar)
{
    int cand, all_cands[MAX_CANDS];

    cand = candsearch_in_pix (targets, num_targets, cent_x, cent_y, dl, dr,
                              du, dd, all_cands, cpar);

    for (cand = 0; cand < MAX_CANDS; cand++) {
        if(all_cands[cand] == -999) {
            reg[cand].ftnr = TR_UNUSED;
        } else {
            reg[cand].whichcam[cam] = 1;
            reg[cand].ftnr = targets[all_cands[cand]].tnr;
        }
    }
}

/* search_volume_center_moving() finds the position of the center of the search
 * volume for a moving particle using the velocity of last step.
 *
 * Arguments:
 * vec3d prev_pos - previous position
 * vec3d curr_pos - current position
 * vec3d *output - output variable, for the  calculated
 *   position.
 */
void search_volume_center_moving(vec3d prev_pos, vec3d curr_pos, vec3d output)
{
    vec_scalar_mul(curr_pos, 2, output);
    vec_subt(output, prev_pos, output);

}

/* predict is used in display loop (only) of track.c to predict the position of
 * a particle in the next frame, using the previous and current positions
 * Arguments:
 * vec2d prev_pos, curr_pos are 2d positions at previous/current frames
 * vec2d output - output of the 2D positions of the particle in the next frame.
 */
void predict (vec2d prev_pos, vec2d curr_pos, vec2d output)
{
    output[0] = 2*curr_pos[0] - prev_pos[0];
    output[1] = 2*curr_pos[1] - prev_pos[1];
}


/* pos3d_in_bounds() checks that all components of a pos3d are in their
   respective bounds taken from a track_par object.

   Arguments:
   vec3d pos - the 3-component array to check.
   track_par *bounds - the struct containing the bounds specification.

   Returns:
   True if all components in bounds, false otherwise.
 */
int pos3d_in_bounds(vec3d pos, track_par *bounds) {
    return (
        bounds->dvxmin < pos[0] && pos[0] < bounds->dvxmax &&
        bounds->dvymin < pos[1] && pos[1] < bounds->dvymax &&
        bounds->dvzmin < pos[2] && pos[2] < bounds->dvzmax );
}

/* angle_acc() calculates the angle between the (1st order) numerical velocity
   vectors to the predicted next position and to the candidate actual position.
   The angle is calculated in [gon], see [1].
   The predicted position is the position if the particle continued at current
   velocity.

   Arguments:
   vec3d start, pred, cand - the particle start position, predicted position,
      and possible actual position, respectively.
   double *angle - output variable, the angle between the two velocity
      vectors, [gon]
   double *acc - output variable, the 1st-order numerical acceleration embodied
      in the deviation from prediction.
 */
void angle_acc(vec3d start, vec3d pred, vec3d cand, double *angle, double *acc)
{
    vec3d v0, v1;

    vec_subt(pred, start, v0);
    vec_subt(cand, start, v1);

    *acc = vec_diff_norm(v0, v1);


    if ((v0[0] == -v1[0]) && (v0[1] == -v1[1]) && (v0[2] == -v1[2])) {
        *angle = 200;
    } else if ((v0[0] == v1[0]) && (v0[1] == v1[1]) && (v0[2] == v1[2])) {
        *angle = 0;         // otherwise it returns NaN
    } else {
        *angle = (200./M_PI) * acos(vec_dot(v0, v1) / vec_norm(v0) \
                                    / vec_norm(v1));
    }
}


/* candsearch_in_pix searches of four (4) near candidates in target list
 * 
 * Arguments:
 * target next[] - array of targets (pointer, x,y, n, nx,ny, sumg, track ID),
 *     assumed to be y sorted.
 * int num_targets - target array length.
 * double cent_x, cent_y - image coordinates of the position of a particle [pixel]
 * double dl, dr, du, dd - respectively the left, right, up, down distance to
 *   the search area borders from its center, [pixel]
 * int p[] - indices in ``next`` of the candidates found.
 * control_par *cpar array of parameters (cpar->imx,imy are needed)
 * 
 * Returns:
 * int, the number of candidates found, between 0 - 3
 */

int candsearch_in_pix (target next[], int num_targets, double cent_x, double cent_y,
                       double dl, double dr, double du, double dd, int p[4], control_par *cpar) {

    int j, j0, dj;
    int counter = 0, p1, p2, p3, p4;
    double d, dmin = 1e20, xmin, xmax, ymin, ymax;
    double d1, d2, d3, d4;

    xmin = cent_x - dl;  xmax = cent_x + dr;  ymin = cent_y - du;  ymax = cent_y + dd;

    if(xmin<0.0) xmin = 0.0;
    if(xmax > cpar->imx)
        xmax = cpar->imx;
    if(ymin<0.0) ymin = 0.0;
    if(ymax > cpar->imy)
        ymax = cpar->imy;

    for (j = 0; j<4; j++) ( p[j] = PT_UNUSED );
    p1 = p2 = p3 = p4 = PT_UNUSED;
    d1 = d2 = d3 = d4 = dmin;


    if (cent_x >= 0.0 && cent_x <= cpar->imx ) {
        if (cent_y >= 0.0 && cent_y <= cpar->imy ) {

            /* binarized search for start point of candidate search */
            for (j0 = num_targets/2, dj = num_targets/4; dj>1; dj /= 2)
            {
                if (next[j0].y < ymin) j0 += dj;
                else j0 -= dj;
            }

            j0 -= 12;  if (j0 < 0) j0 = 0;             /* due to trunc */
            for (j = j0; j<num_targets; j++) {           /* candidate search */
                if (next[j].tnr != TR_UNUSED ) {
                    if (next[j].y > ymax ) break;                     /* finish search */
                    if (next[j].x > xmin && next[j].x < xmax \
                        && next[j].y > ymin && next[j].y < ymax) {
                        d = sqrt ((cent_x-next[j].x)*(cent_x-next[j].x) + \
                                  (cent_y-next[j].y)*(cent_y-next[j].y));

                        if (d < dmin) {
                            dmin = d;
                        }

                        if ( d < d1 ) {
                            p4 = p3; p3 = p2; p2 = p1; p1 = j;
                            d4 = d3; d3 = d2; d2 = d1; d1 = d;
                        }
                        else if ( d1 < d &&  d < d2 ) {
                            p4 = p3; p3 = p2; p2 = j;
                            d4 = d3; d3 = d2; d2 = d;
                        }
                        else if ( d2 < d && d < d3 ) {
                            p4 = p3; p3 = j;
                            d4 = d3; d3 = d;
                        }
                        else if ( d3 < d && d < d4 ) {
                            p4 = j;
                            d4 = d;
                        }
                    }
                }
            }

            p[0] = p1;
            p[1] = p2;
            p[2] = p3;
            p[3] = p4;

            for (j = 0; j<4; j++) if ( p[j] != PT_UNUSED ) counter++;
        }         /* if x is within the image boundaries */
    }     /* if y is within the image boundaries */
    return (counter);
}

/* candsearch_in_pix_rest searches for a nearest candidate in unmatched target list	
 * 	
 * Arguments:	
 * target next[] - array of targets (pointer, x,y, n, nx,ny, sumg, track ID),	
 *     assumed to be y sorted.	
 * int num_targets - target array length.	
 * double cent_x, cent_y - image coordinates of the position of a particle [pixel]	
 * double dl, dr, du, dd - respectively the left, right, up, down distance to	
 *   the search area borders from its center, [pixel]	
 * int p[] - indices in ``next`` of the candidates found.	
 * control_par *cpar array of parameters (cpar->imx,imy are needed)	
 * 	
 * Returns:	
 * int, the number of candidates found, between 0 - 1	
 */	

int candsearch_in_pix_rest (target next[], int num_targets, double cent_x, double cent_y,	
                       double dl, double dr, double du, double dd, int p[], control_par *cpar) {	

    int j, j0, dj;	
    int counter = 0;	
    double d, dmin = 1e20, xmin, xmax, ymin, ymax;	
    // double d1, d2, d3, d4;	

    xmin = cent_x - dl;  xmax = cent_x + dr;  ymin = cent_y - du;  ymax = cent_y + dd;	

    if(xmin<0.0) xmin = 0.0;	
    if(xmax > cpar->imx)	
        xmax = cpar->imx;	
    if(ymin<0.0) ymin = 0.0;	
    if(ymax > cpar->imy)	
        ymax = cpar->imy;	

    p[0] = PT_UNUSED;	

    if (cent_x >= 0.0 && cent_x <= cpar->imx ) {	
        if (cent_y >= 0.0 && cent_y <= cpar->imy ) {	

            /* binarized search for start point of candidate search */	
            for (j0 = num_targets/2, dj = num_targets/4; dj>1; dj /= 2)	
            {	
                if (next[j0].y < ymin) j0 += dj;	
                else j0 -= dj;	
            }	

            j0 -= 12;  if (j0 < 0) j0 = 0;             /* due to trunc */	
            for (j = j0; j<num_targets; j++) {           /* candidate search */	
                if (next[j].tnr == TR_UNUSED ) {	
                    if (next[j].y > ymax ) break;                     /* finish search */	
                    if (next[j].x > xmin && next[j].x < xmax \
                        && next[j].y > ymin && next[j].y < ymax) {	
                        d = sqrt ((cent_x-next[j].x)*(cent_x-next[j].x) + \
                                  (cent_y-next[j].y)*(cent_y-next[j].y));	

                        if (d < dmin) {	
                            dmin = d;	
                            p[0] = j;	
                        }   	
                    }	
                }	
            }			

            if ( p[0] != PT_UNUSED ) counter++;	
        }  /* if y is within the image boundaries */	
    }     /* if x is within the image boundaries */	
    return (counter);	
}

/* searchquader defines the search region, using tracking parameters
 * dvxmin, ... dvzmax (but within the image boundaries), per camera
 * Its primary objective is to provide a safe search region in each camera
 * to the following candidate search in pixels (candsearch_in_pix).
 * We project a position of a center of search from 3D to the image space (pixels)
 * and project also 8 corners in 3D of a cuboid on the image space
 * The searchquader returns the distances from the center of search to the
 * left, right, up and down, in any case not crossing the image size limits
 * (0,0) and (cpar->imx, cpar->imy). If the point is on the border, the search
 * region is only in the allowed directions.
 * Arguments:
 * vec3d point position in physical space
 * track_par *tpar set of tracking parameters
 * control_par *cpar set of control parameters for the num_cams
 * Calibration *cal calibration per camera to find a projection of a 3D vertex
 * of a cuboid in the image space.
 * Returns the arrays xr,xl,yd,yu (right, left, down, up) per camera
 * for the search of a quader (cuboid), given in pixel distances, relative to the
 * point of search.
 */
void searchquader(vec3d point, double xr[4], double xl[4], double yd[4], \
                  double yu[4], track_par *tpar, control_par *cpar, Calibration **cal){
    int i, pt, dim;
    vec3d mins, maxes;
    vec2d corner, center;
    vec3d quader[8];



    vec_set(mins, tpar->dvxmin, tpar->dvymin, tpar->dvzmin);
    vec_set(maxes, tpar->dvxmax, tpar->dvymax, tpar->dvzmax);

    /* 3D positions of search volume - eight corners of a box */
    for (pt = 0; pt < 8; pt++) {
        vec_copy(quader[pt], point);
        for (dim = 0; dim < 3; dim++) {
            if (pt & 1<<dim) {
                quader[pt][dim] += maxes[dim];
            } else {
                quader[pt][dim] += mins[dim];
            }
        }
    }


    /* calculation of search area in each camera */
    for (i = 0; i < cpar->num_cams; i++) {

        /* initially large or small values */
        xr[i] = 0;
        xl[i] = cpar->imx;
        yd[i] = 0;
        yu[i] = cpar->imy;

        /* pixel position of a search center */
        point_to_pixel (center, point, cal[i], cpar);

        /* mark 4 corners of the search region in pixels */
        for (pt = 0; pt < 8; pt++) {
            point_to_pixel (corner, quader[pt], cal[i], cpar);

            if (corner[0] < xl[i] ) xl[i] = corner[0];
            if (corner[1] < yu[i] ) yu[i] = corner[1];
            if (corner[0] > xr[i] ) xr[i] = corner[0];
            if (corner[1] > yd[i] ) yd[i] = corner[1];
        }

        if (xl[i] < 0 ) xl[i] = 0;
        if (yu[i] < 0 ) yu[i] = 0;
        if (xr[i] > cpar->imx)
            xr[i] = cpar->imx;
        if (yd[i] > cpar->imy)
            yd[i] = cpar->imy;

        /* eventually xr,xl,yd,yu are pixel distances relative to the point */
        xr[i] = xr[i]     - center[0];
        xl[i] = center[0] - xl[i];
        yd[i] = yd[i]     - center[1];
        yu[i] = center[1] - yu[i];
    }
}

/* sort_candidates_by_freq() sorts (in place!) the list of candidates in 
 * foundpix array by frequency of their appearance in all the cameras.
 *
 * Arguments:
 * foundpix item[] - as long as num_cam*MAX_CANDS candidates. Sorted in place!
 * int num_cams - number of cameras in the experiment (typically 1-4)
 * Returns the sorted array foundpix item[] and a pointer *counter to
 * an integer number of different candidates
 *
 * Retuens:
 * The number of distinct particles in the sorted array.
 */

int sort_candidates_by_freq(foundpix item[], int num_cams) {
    int i,j,m, different;
    foundpix temp;

    different = 0;

    /* where what was found */
    for (i = 0; i<num_cams*MAX_CANDS; i++)
        for (j = 0; j<num_cams; j++)
            for (m = 0; m<MAX_CANDS; m++)
                if(item[i].ftnr == item[4*j+m].ftnr)
                {
                    item[i].whichcam[j] = 1;
                }

    /* how often was ftnr found */
    for (i = 0; i<num_cams*MAX_CANDS; i++)
        for (j = 0; j < num_cams; j++)
            if (item[i].whichcam[j] == 1 && item[i].ftnr != TR_UNUSED) {
                item[i].freq++;
            }

    /* sort freq */
    for (i = 1; i<num_cams*MAX_CANDS; ++i) for (j = num_cams*MAX_CANDS-1; j>=i; --j)
        {
            if ( item[j-1].freq < item[j].freq )
            {
                temp = *(item+j-1); *(item+j-1) = *(item+j); *(item+j) = temp;
            }
        }

    /* prune the duplicates or those that are found only once */
    for (i = 0; i<num_cams*MAX_CANDS; i++)
        for (j = i+1; j<num_cams*MAX_CANDS; j++)
        {
            if (item[i].ftnr == item[j].ftnr || item[j].freq <2)
            {
                item[j].freq = 0;
                item[j].ftnr = TR_UNUSED;
            }
        }

    /* sort freq again on the clean dataset */
    for (i = 1; i<num_cams*MAX_CANDS; ++i) for (j = num_cams*MAX_CANDS-1; j>=i; --j)
        {
            if ( item[j-1].freq < item[j].freq )
            {
                temp = *(item+j-1); *(item+j-1) = *(item+j); *(item+j) = temp;
            }
        }

    for (i = 0; i<num_cams*MAX_CANDS; ++i) if(item[i].freq != 0) different++;
    return different;
}

/* sorts a float array a and an integer array b both of length n
 * Arguments:
 *  float array a (returned sorted in the ascending order)
 *  integer array b (returned sorted according to float array a)
 *  int n (length of a)
 */
void sort(int n, float a[], int b[]){
    int flag = 0, i, itemp;
    float ftemp;

    do {
        flag = 0;
        for(i = 0; i<(n-1); i++)
            if(a[i] > a[i+1]) {
                ftemp = a[i];
                itemp = b[i];
                a[i] = a[i+1];
                b[i] = b[i+1];
                a[i+1] = ftemp;
                b[i+1] = itemp;
                flag = 1;
            }
    } while(flag);
}

/* point_to_pixel is just a shortcut to two lines that repeat every so in track loop
 * img_coord (from 3d point to a 2d vector in metric units), followed by
 * metric_to_pixel (from 2d vector in metric units to the pixel position in the camera)
 * Arguments:
 * vec3d point in 3D space
 * Calibration *cal parameters
 * Control parameters (num cams, multimedia parameters, cpar->mm, etc.)
 * Returns (as a first argument):
 * vec2d with pixel positions (x,y) in the camera.
 */
void point_to_pixel (vec2d v1, vec3d point, Calibration *cal, control_par *cpar){
    img_coord(point, cal, cpar->mm, &v1[0], &v1[1]);
    metric_to_pixel(&v1[0], &v1[1], v1[0], v1[1], cpar);
}

/* sorted_candidates_in_volume() receives a volume center and produces a list
   of candidates for the next particle in that volume, sorted by the 
   candidates' number of appearances as 2D targets.
   
   Arguments:
   vec3d center - the 3D midpoint-position of the search volume
   vec2d center_proj[] - projections of the center on the cameras, pixel 
      coordinates.
   frame *frm - the frame holding targets for the search.
   tracking_run *run - the parameter collection we need for determining 
        search region. The same object used throughout the tracking code.
   
   Returns:
   foundpix *points - a newly-allocated buffer of foundpix items, denoting
      for each item its particle number and quality parameters. The buffer
      is terminated by one extra item with ftnr set to TR_UNUSED
*/
foundpix *sorted_candidates_in_volume(vec3d center, vec2d center_proj[],
    frame *frm, tracking_run *run)
{
    foundpix *points;
    double right[TR_MAX_CAMS], left[TR_MAX_CAMS];
    double down[TR_MAX_CAMS], up[TR_MAX_CAMS];
    int cam, num_cams, num_cands;
    
    num_cams = frm->num_cams;
    points = (foundpix*) calloc(num_cams*MAX_CANDS, sizeof(foundpix));
    reset_foundpix_array(points, num_cams*MAX_CANDS, num_cams);
    
    /* Search limits in image space */
    searchquader(center, right, left, down, up, 
        run->tpar, run->cpar, run->cal);
    
    /* search in pix for candidates in the next time step */
    for (cam = 0; cam < num_cams; cam++) {
        register_closest_neighbs(frm->targets[cam], 
            frm->num_targets[cam], cam, 
            center_proj[cam][0], center_proj[cam][1],
            left[cam], right[cam], up[cam], down[cam], 
            &(points[cam*MAX_CANDS]), run->cpar
        );
    }

    /* fill and sort candidate struct */
    num_cands = sort_candidates_by_freq(points, num_cams);
    if (num_cands > 0) {
        points = (foundpix *) realloc(
            points, (num_cands + 1)*sizeof(foundpix));
        points[num_cands].ftnr = TR_UNUSED;
        return points;
    } else {
        free(points);
        return NULL;
    }
}

/* asses_new_position() determines the nearest target on each camera around a 
 * search position and prepares the data structures accordingly with the 
 * determined target info or the unused flag value.
 * 
 * Arguments:
 * vec3d pos - the position around which to search.
 * vec2d targ_pos[] - the determined targets' respective positions.
 * int cand_inds[][MAX_CANDS] - output buffer, the determined targets' index in
 *    the respective camera's target list
 * frame *frm - the frame holdin target data for the search position.
 * tracking_run *run - scene information struct.
 * 
 * Returns:
 * the number of cameras where a suitable target was found.
 */
int assess_new_position(vec3d pos, vec2d targ_pos[], 
    int cand_inds[][MAX_CANDS], frame *frm, tracking_run *run) 
{
    int cam, num_cands, valid_cams, _ix;
    vec2d pixel;
    double right, left, down, up; /* search rectangle limits */
    
    left = right = up = down = ADD_PART;

    for (cam = 0; cam < TR_MAX_CAMS; cam++) {
        targ_pos[cam][0] = targ_pos[cam][1] = COORD_UNUSED;
    }

    for (cam = 0; cam < run->cpar->num_cams; cam++) {
        point_to_pixel(pixel, pos, run->cal[cam], run->cpar);
        
        /* here we shall use only the 1st neigbhour */
        num_cands = candsearch_in_pix_rest (frm->targets[cam], frm->num_targets[cam],
            pixel[0], pixel[1], left, right, up, down, 
            cand_inds[cam], run->cpar);
        
        // printf("num_cands after pix_rest is %d\n",num_cands);

        if (num_cands > 0) {
            _ix = cand_inds[cam][0];  // first nearest neighbour
            targ_pos[cam][0] = frm->targets[cam][_ix].x;
            targ_pos[cam][1] = frm->targets[cam][_ix].y;
        }
    }

    valid_cams = 0;
    for (cam = 0; cam < run->cpar->num_cams; cam++) {
        if ((targ_pos[cam][0] != COORD_UNUSED) && \
            (targ_pos[cam][1] != COORD_UNUSED)) 
        {
            pixel_to_metric(&(targ_pos[cam][0]), &(targ_pos[cam][1]), 
                targ_pos[cam][0], targ_pos[cam][1], run->cpar);
            dist_to_flat(targ_pos[cam][0], targ_pos[cam][1], run->cal[cam], 
                &(targ_pos[cam][0]), &(targ_pos[cam][1]), run->flatten_tol);
            valid_cams++;
        }
    }
    return valid_cams;
}

/* add_particle() inserts a particle at a given position to the end of the 
 * frame, along with associated targets.
 * 
 * Arguments:
 * frame *frm - the frame to store the particle.
 * vec3d pos - position of inserted particle in the global coordinates.
 * int cand_inds[][MAX_CANDS] - indices of candidate targets for association
 *    with this particle.
 */
void add_particle(frame *frm, vec3d pos, int cand_inds[][MAX_CANDS]) {
    int num_parts, cam, _ix;
    P *ref_path_inf;
    corres *ref_corres;
    target **ref_targets;
    
    num_parts = frm->num_parts;
    ref_path_inf = &(frm->path_info[num_parts]);
    vec_copy(ref_path_inf->x, pos);
    reset_links(ref_path_inf);
    
    ref_corres = &(frm->correspond[num_parts]);
    ref_targets = frm->targets;
    for (cam = 0; cam < frm->num_cams; cam++) {
        ref_corres->p[cam] = CORRES_NONE;
        
        /* We always take the 1st candidate, apparently. Why did we fetch 4? */
        if(cand_inds[cam][0] != PT_UNUSED) {
            _ix = cand_inds[cam][0];
            ref_targets[cam][_ix].tnr = num_parts;
            ref_corres->p[cam] = _ix;
            ref_corres->nr = num_parts;
        }
    }
    frm->num_parts++;
}

/* trackcorr_c_loop is the main tracking subroutine that scans the 3D particle position
 * data from rt_is.* files and the 2D particle positions in image space in _targets and
 * constructs trajectories (links) of the particles in 3D in time.
 * the basic concepts of the tracking procedure are from the following publication by
 * Jochen Willneff: "A New Spatio-Temporal Matching Algorithm For 3D-Particle Tracking Velocimetry"
 * https://www.mendeley.com/catalog/new-spatiotemporal-matching-algorithm-3dparticle-tracking-velocimetry/
 * or http://e-collection.library.ethz.ch/view/eth:26978
 * this method is an extension of the previously used tracking method described in details in
 * Malik et al. 1993: "Particle tracking velocimetry in three-dimensional flows: Particle tracking"
 * http://mnd.ly/2dCt3um
 *
 * Arguments:
 * tracking_run *run_info pointer to the (sliding) frame dataset of 4 frames of particle positions
 * and all the needed parameters underneath: control, volume, etc.
 * integer step number or the frame number from the sequence
 * Note: step is not really setting up the step to track, the buffer provided to the trackcoor_c_loop
 * is already preset by 4 frames buf[0] to buf[3] and we track particles in buf[1], i.e. one "previous"
 * one present and two future frames.
 * 
 * Returns: function does not return an argument, the tracks are updated within the run_info dataset
 */
void trackcorr_c_loop (tracking_run *run_info, int step) {
    /* sequence loop */
    int j, h, mm, kk, in_volume = 0;
    int philf[4][MAX_CANDS];
    int count1 = 0, count2 = 0, count3 = 0, num_added = 0;
    int quali = 0;
    vec3d diff_pos, X[6];     /* 7 reference points used in the algorithm, TODO: check if can reuse some */
    double angle, acc, angle0, acc0,  dl;
    double angle1, acc1;
    vec2d v1[4], v2[4]; /* volume center projection on cameras */ 
    double rr;


    /* Shortcuts to inside current frame */
    P *curr_path_inf, *ref_path_inf;
    corres *curr_corres;
    target **curr_targets;
    int _ix;     /* For use in any of the complex index expressions below */
    int orig_parts; /* avoid infinite loop with particle addition set */

    /* Shortcuts into the tracking_run struct */ 
    Calibration **cal;
    framebuf_base *fb;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;

    foundpix *w, *wn; 
    count1 = 0; num_added = 0;

    fb = run_info->fb;
    cal = run_info->cal;
    tpar = run_info->tpar;
    vpar = run_info->vpar;
    cpar = run_info->cpar;
    curr_targets = fb->buf[1]->targets;

    /* try to track correspondences from previous 0 - corp, variable h */
    orig_parts = fb->buf[1]->num_parts;
    for (h = 0; h < orig_parts; h++) {
        for (j = 0; j < 6; j++) vec_init(X[j]);

        curr_path_inf = &(fb->buf[1]->path_info[h]);
        curr_corres = &(fb->buf[1]->correspond[h]);

        curr_path_inf->inlist = 0;

        /* 3D-position */
        vec_copy(X[1], curr_path_inf->x);

        /* use information from previous to locate new search position
           and to calculate values for search area */
        if (curr_path_inf->prev >= 0) {
            ref_path_inf = &(fb->buf[0]->path_info[curr_path_inf->prev]);
            vec_copy(X[0], ref_path_inf->x);
            search_volume_center_moving(ref_path_inf->x, curr_path_inf->x, X[2]);

            for (j = 0; j < fb->num_cams; j++) {
                point_to_pixel (v1[j], X[2], cal[j], cpar);
            }
        } else {
            vec_copy(X[2], X[1]);
            for (j = 0; j < fb->num_cams; j++) {
                if (curr_corres->p[j] == CORRES_NONE) {
                    point_to_pixel (v1[j], X[2], cal[j], cpar);
                } else {
                    _ix = curr_corres->p[j];
                    v1[j][0] = curr_targets[j][_ix].x;
                    v1[j][1] = curr_targets[j][_ix].y;
                }
            }

        }

        /* calculate search cuboid and reproject it to the image space */
        w = sorted_candidates_in_volume(X[2], v1, fb->buf[2], run_info);
        if (w == NULL) continue;

        /* Continue to find candidates for the candidates. */
        count2++;
        mm = 0;
        while (w[mm].ftnr != TR_UNUSED) {       /* counter1-loop */
            /* search for found corr of current the corr in next
               with predicted location */

            /* found 3D-position */
            ref_path_inf = &(fb->buf[2]->path_info[w[mm].ftnr]);
            vec_copy(X[3], ref_path_inf->x);

            if (curr_path_inf->prev >= 0) {
                for (j = 0; j < 3; j++)
                    X[5][j] = 0.5*(5.0*X[3][j] - 4.0*X[1][j] + X[0][j]);
            } else {
                search_volume_center_moving(X[1], X[3], X[5]);
            }

            for (j = 0; j < fb->num_cams; j++) {
                point_to_pixel (v1[j], X[5], cal[j], cpar);
            }

            /* end of search in pix */
            wn = sorted_candidates_in_volume(X[5], v1, fb->buf[3], run_info);
            if (wn != NULL) {
                count3++;
                kk = 0;
                while (wn[kk].ftnr != TR_UNUSED) {
                    ref_path_inf = &(fb->buf[3]->path_info[wn[kk].ftnr]);
                    vec_copy(X[4], ref_path_inf->x);

                    vec_subt(X[4], X[3], diff_pos);
                    if ( pos3d_in_bounds(diff_pos, tpar)) {
                        angle_acc(X[3], X[4], X[5], &angle1, &acc1);
                        if (curr_path_inf->prev >= 0) {
                            angle_acc(X[1], X[2], X[3], &angle0, &acc0);
                        } else {
                            acc0 = acc1; angle0 = angle1;
                        }

                        acc = (acc0+acc1)/2; angle = (angle0+angle1)/2;
                        quali = wn[kk].freq+w[mm].freq;

                        if ((acc < tpar->dacc && angle < tpar->dangle) || \
                            (acc < tpar->dacc/10))
                        {
                            dl = (vec_diff_norm(X[1], X[3]) +
                                vec_diff_norm(X[4], X[3]) )/2;
                            rr = (dl/run_info->lmax + acc/tpar->dacc + \
                                angle/tpar->dangle)/(quali);
                            register_link_candidate(
                                curr_path_inf, rr, w[mm].ftnr);
                        }
                    }
                    kk++;
                } /* End of searching 2nd-frame candidates. */
            }             

            /* creating new particle position,
             *  reset img coord because of num_cams < 4
             *  fix distance of 3 pixels to define xl,xr,yu,yd instead of searchquader
             *  and search for unused candidates in next time step
             */
            quali = assess_new_position(X[5], v2, philf, fb->buf[3], run_info); 
                        
            /* quali >=2 means at least in two cameras
             * we found a candidate
             */
            if ( quali >= 2) {
                in_volume = 0;                 //inside volume

                dl = point_position(v2, cpar->num_cams, cpar->mm, cal, X[4]);

                /* volume check */
                if ( vpar->X_lay[0] < X[4][0] && X[4][0] < vpar->X_lay[1] &&
                     run_info->ymin < X[4][1] && X[4][1] < run_info->ymax &&
                     vpar->Zmin_lay[0] < X[4][2] && X[4][2] < vpar->Zmax_lay[1])
                {
                    in_volume = 1;
                }

                vec_subt(X[3], X[4], diff_pos);
                if ( in_volume == 1 && pos3d_in_bounds(diff_pos, tpar) ) {
                    angle_acc(X[3], X[4], X[5], &angle, &acc);

                    if ((acc < tpar->dacc && angle < tpar->dangle) || \
                        (acc < tpar->dacc/10))
                    {
                        dl = (vec_diff_norm(X[1], X[3]) +
                              vec_diff_norm(X[4], X[3]) )/2;
                        rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle) /
                             (quali+w[mm].freq);
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr);

                        if (tpar->add) {
                            add_particle(fb->buf[3], X[4], philf);
                            num_added++;
                        }
                    }
                }
                in_volume = 0;
            }
            quali = 0;

            /* end of creating new particle position */
            /* *************************************************************** */

            /* try to link if kk is not found/good enough and prev exist */
            if ( curr_path_inf->inlist == 0 && curr_path_inf->prev >= 0 ) {
                vec_subt(X[3], X[1], diff_pos);

                if (pos3d_in_bounds(diff_pos, tpar)) {
                    angle_acc(X[1], X[2], X[3], &angle, &acc);

                    if ( (acc < tpar->dacc && angle < tpar->dangle) || \
                         (acc < tpar->dacc/10) )
                    {
                        quali = w[mm].freq;
                        dl = (vec_diff_norm(X[1], X[3]) +
                              vec_diff_norm(X[0], X[1]) )/2;
                        rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle)/(quali);
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr);
                    }
                }
            }

            free(wn);
            mm++;
        } /* end of loop over first-frame candidates. */

        /* begin of inlist still zero */
        if (tpar->add) {
            if ( curr_path_inf->inlist == 0 && curr_path_inf->prev >= 0 ) {
                quali = assess_new_position(X[2], v2, philf, fb->buf[2], run_info);

                if (quali>=2) {
                    vec_copy(X[3], X[2]);
                    in_volume = 0;

                    dl = point_position(v2, fb->num_cams, cpar->mm, cal, X[3]);

                    /* in volume check */
                    if ( vpar->X_lay[0] < X[3][0] && X[3][0] < vpar->X_lay[1] &&
                         run_info->ymin < X[3][1] && X[3][1] < run_info->ymax &&
                         vpar->Zmin_lay[0] < X[3][2] &&
                         X[3][2] < vpar->Zmax_lay[1])
                    {
                        in_volume = 1;
                    }

                    vec_subt(X[2], X[3], diff_pos);
                    if ( in_volume == 1 && pos3d_in_bounds(diff_pos, tpar) ) {
                        angle_acc(X[1], X[2], X[3], &angle, &acc);

                        if ( (acc < tpar->dacc && angle < tpar->dangle) || \
                             (acc < tpar->dacc/10) )
                        {
                            dl = (vec_diff_norm(X[1], X[3]) +
                                  vec_diff_norm(X[0], X[1]) )/2;
                            rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle)/(quali);
                            register_link_candidate(curr_path_inf, rr, fb->buf[2]->num_parts);

                            add_particle(fb->buf[2], X[3], philf);
                            num_added++;
                        }
                    }
                    in_volume = 0;
                }                 // if quali >= 2
            }
        }
        /* end of inlist still zero */
        /***********************************/

        free(w);
    }     /* end of h-loop */

    /* sort decis and give preliminary "finaldecis"  */
    for (h = 0; h < fb->buf[1]->num_parts; h++) {
        curr_path_inf = &(fb->buf[1]->path_info[h]);

        if(curr_path_inf->inlist > 0 ) {
            sort(curr_path_inf->inlist, (float *) curr_path_inf->decis,
                 curr_path_inf->linkdecis);
            curr_path_inf->finaldecis = curr_path_inf->decis[0];
            curr_path_inf->next = curr_path_inf->linkdecis[0];
        }
    }

    /* create links with decision check */
    for (h = 0; h < fb->buf[1]->num_parts; h++) {
        curr_path_inf = &(fb->buf[1]->path_info[h]);

        if(curr_path_inf->inlist > 0 ) {
            ref_path_inf = &(fb->buf[2]->path_info[curr_path_inf->next]);

            if (ref_path_inf->prev == PREV_NONE) {
                /* best choice wasn't used yet, so link is created */
                ref_path_inf->prev = h;
            } else {
                /* best choice was already used by mega[2][mega[1][h].next].prev */
                /* check which is the better choice */
                if ( fb->buf[1]->path_info[ref_path_inf->prev].finaldecis > \
                     curr_path_inf->finaldecis)
                {
                    /* remove link with prev */
                    fb->buf[1]->path_info[ref_path_inf->prev].next = NEXT_NONE;
                    ref_path_inf->prev = h;
                } else {
                    curr_path_inf->next = NEXT_NONE;
                }
            }
        }
        if (curr_path_inf->next != NEXT_NONE ) count1++;
    }
    /* end of creation of links with decision check */

    printf ("step: %d, curr: %d, next: %d, links: %d, lost: %d, add: %d\n",
            step, fb->buf[1]->num_parts, fb->buf[2]->num_parts, count1,
            fb->buf[1]->num_parts - count1, num_added);

    /* for the average of particles and links */
    run_info->npart = run_info->npart + fb->buf[1]->num_parts;
    run_info->nlinks = run_info->nlinks + count1;

    fb_next(fb);
    fb_write_frame_from_start(fb, step);
    if(step < run_info->seq_par->last - 2) {
        fb_read_frame_at_end(fb, step + 3, 0);
    }
} /* end of sequence loop */

void trackcorr_c_finish(tracking_run *run_info, int step)
{
    int range = run_info->seq_par->last - run_info->seq_par->first;
    double npart, nlinks;

    /* average of all steps */
    npart = (double)run_info->npart / range;
    nlinks = (double)run_info->nlinks / range;
    printf ("Average over sequence, particles: %5.1f, links: %5.1f, lost: %5.1f\n",
            npart, nlinks, npart - nlinks);

    fb_next(run_info->fb);
    fb_write_frame_from_start(run_info->fb, step);
}

/*     track backwards */
double trackback_c (tracking_run *run_info)
{
    int i, j, h, in_volume = 0;
    int step;
    int philf[4][MAX_CANDS];
    int count1 = 0, count2 = 0, num_added = 0;
    int quali = 0;
    double angle, acc, dl;
    vec3d diff_pos, X[6];     /* 6 reference points used in the algorithm */
    vec2d n[4], v2[4];     // replaces xn,yn, x2[4], y2[4],
    double rr, Ymin = 0, Ymax = 0;
    double npart = 0, nlinks = 0;
    foundpix *w;

    sequence_par *seq_par;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;
    framebuf_base *fb;
    Calibration **cal;

    /* Shortcuts to inside current frame */
    P *curr_path_inf, *ref_path_inf;

    /* shortcuts */
    cal = run_info->cal;
    seq_par = run_info->seq_par;
    tpar = run_info->tpar;
    vpar = run_info->vpar;
    cpar = run_info->cpar;

    fb = run_info->fb;

    /* Prime the buffer with first frames */
    for (step = seq_par->last; step > seq_par->last - 4; step--) {
        fb_read_frame_at_end(fb, step, 1);
        fb_next(fb);
    }
    fb_prev(fb);

    /* sequence loop */
    for (step = seq_par->last - 1; step > seq_par->first; step--) {
        // printf ("Time step: %d, seqnr: %d:\n",
        //         step - seq_par->first, step);

        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);

            /* We try to find link only if the forward search failed to. */
            if ((curr_path_inf->next < 0) || (curr_path_inf->prev != -1)) continue;

            for (j = 0; j < 6; j++) vec_init(X[j]);
            curr_path_inf->inlist = 0;

            /* 3D-position of current particle */
            vec_copy(X[1], curr_path_inf->x);

            /* use information from previous to locate new search position
               and to calculate values for search area */
            ref_path_inf = &(fb->buf[0]->path_info[curr_path_inf->next]);
            vec_copy(X[0], ref_path_inf->x);
            search_volume_center_moving(ref_path_inf->x, curr_path_inf->x, X[2]);
            
            for (j = 0; j < fb->num_cams; j++) {
                point_to_pixel (n[j], X[2], cal[j], cpar);
            }
            
            /* calculate searchquader and reprojection in image space */
            w = sorted_candidates_in_volume(X[2], n, fb->buf[2], run_info);
            
            if (w != NULL) {
                count2++;

                i = 0;
                while (w[i].ftnr != TR_UNUSED) {
                    ref_path_inf = &(fb->buf[2]->path_info[w[i].ftnr]);
                    vec_copy(X[3], ref_path_inf->x);

                    vec_subt(X[1], X[3], diff_pos);
                    if (pos3d_in_bounds(diff_pos, tpar)) {
                        angle_acc(X[1], X[2], X[3], &angle, &acc);

                        /* *********************check link *****************************/
                        if ((acc < tpar->dacc && angle < tpar->dangle) || \
                            (acc < tpar->dacc/10))
                        {
                            dl = (vec_diff_norm(X[1], X[3]) +
                                  vec_diff_norm(X[0], X[1]) )/2;
                            quali = w[i].freq;
                            rr = (dl/run_info->lmax + acc/tpar->dacc + \
                                angle/tpar->dangle)/quali;
                            register_link_candidate(curr_path_inf, rr, w[i].ftnr);
                        }
                    }
                    i++;
                }
            }

            free(w);

            /* if old wasn't found try to create new particle position from rest */
            if (tpar->add) {
                if ( curr_path_inf->inlist == 0) {
                    quali = assess_new_position(X[2], v2, philf, fb->buf[2], run_info);
                    if (quali>=2) {
                        //vec_copy(X[3], X[2]);
                        in_volume = 0;

                        point_position(v2, fb->num_cams, cpar->mm, cal, X[3]);

                        /* volume check */
                        if ( vpar->X_lay[0] < X[3][0] && X[3][0] < vpar->X_lay[1] &&
                             Ymin < X[3][1] && X[3][1] < Ymax &&
                             vpar->Zmin_lay[0] < X[3][2] && X[3][2] < vpar->Zmax_lay[1])
                        {in_volume = 1;}

                        vec_subt(X[1], X[3], diff_pos);
                        if (in_volume == 1 && pos3d_in_bounds(diff_pos, tpar)) {
                            angle_acc(X[1], X[2], X[3], &angle, &acc);

                            if ( (acc<tpar->dacc && angle<tpar->dangle) || \
                                 (acc<tpar->dacc/10) )
                            {
                                dl = (vec_diff_norm(X[1], X[3]) +
                                      vec_diff_norm(X[0], X[1]) )/2;
                                rr = (dl/run_info->lmax+acc/tpar->dacc + angle/tpar->dangle)/(quali);
                                register_link_candidate(curr_path_inf, rr, fb->buf[2]->num_parts);

                                add_particle(fb->buf[2], X[3], philf);
                            }
                        }
                        in_volume = 0;
                    }
                }
            }             /* end of if old wasn't found try to create new particle position from rest */
        }         /* end of h-loop */

        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);

            if(curr_path_inf->inlist > 0 ) {
                sort(curr_path_inf->inlist, (float *)curr_path_inf->decis,
                     curr_path_inf->linkdecis);
            }
        }

        /* create links with decision check */
        count1 = 0; num_added = 0;
        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);

            if (curr_path_inf->inlist > 0 ) {
                /* if old/new and unused prev == -1 and next == -2 link is created */
                ref_path_inf = &(fb->buf[2]->path_info[curr_path_inf->linkdecis[0]]);

                if ( ref_path_inf->prev == PREV_NONE && \
                     ref_path_inf->next == NEXT_NONE )
                {
                    curr_path_inf->finaldecis = curr_path_inf->decis[0];
                    curr_path_inf->prev = curr_path_inf->linkdecis[0];
                    fb->buf[2]->path_info[curr_path_inf->prev].next = h;
                    num_added++;
                }

                /* old which link to prev has to be checked */
                if ((ref_path_inf->prev != PREV_NONE) && \
                    (ref_path_inf->next == NEXT_NONE) )
                {
                    vec_copy(X[0], fb->buf[0]->path_info[curr_path_inf->next].x);
                    vec_copy(X[1], curr_path_inf->x);
                    vec_copy(X[3], ref_path_inf->x);
                    vec_copy(X[4], fb->buf[3]->path_info[ref_path_inf->prev].x);
                    for (j = 0; j < 3; j++)
                        X[5][j] = 0.5*(5.0*X[3][j] - 4.0*X[1][j] + X[0][j]);

                    angle_acc(X[3], X[4], X[5], &angle, &acc);

                    if ( (acc<tpar->dacc && angle<tpar->dangle) ||  (acc<tpar->dacc/10) ) {
                        curr_path_inf->finaldecis = curr_path_inf->decis[0];
                        curr_path_inf->prev = curr_path_inf->linkdecis[0];
                        fb->buf[2]->path_info[curr_path_inf->prev].next = h;
                        num_added++;
                    }
                }
            }

            if (curr_path_inf->prev != PREV_NONE ) count1++;
        }         /* end of creation of links with decision check */


        printf ("step: %d, curr: %d, next: %d, links: %d, lost: %d, add: %d \n",
                step, fb->buf[1]->num_parts, fb->buf[2]->num_parts, count1,
                fb->buf[1]->num_parts - count1, num_added);

        /* for the average of particles and links */
        npart = npart + fb->buf[1]->num_parts;
        nlinks = nlinks + count1;

        fb_next(fb);
        fb_write_frame_from_start(fb, step);
        if(step > seq_par->first + 2) { fb_read_frame_at_end(fb, step - 3, 1); }
    }     /* end of sequence loop */

    /* average of all steps */
    npart /= (seq_par->last - seq_par->first - 1);
    nlinks /= (seq_par->last - seq_par->first - 1);

    printf ("Average over sequence, particles: %5.1f, links: %5.1f, lost: %5.1f\n",
            npart, nlinks, npart-nlinks);

    fb_next(fb);
    fb_write_frame_from_start(fb, step);

    return nlinks;
}
