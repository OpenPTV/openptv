/* routines for determining the camera parameters used in a scene.
 *
 * References:
 * [1] Maas, Gruen and Papantoniou. Particle tracking velocimetry in 
 *     three-dimensional flows, Part 1: 
 *     Photogrammetric determination of particle coordinates. Experiments in 
 *     Fluids, vol. 15, pp. 133-146, 1993.     
 */

#include "orientation.h"
#include "calibration.h"
#include "ray_tracing.h"
#include "vec_utils.h"
#include "sortgrid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_ITER  80
#define POS_INF 1E20
#define CONVERGENCE 0.00001

/*  skew_midpoint() finds the middle of the minimal distance segment between
    skew rays. Undefined for parallel rays.

    Reference for algorithm:
    docs/skew_midpoint.pdf

    Arguments:
    vec3d vert1, direct1 - vertex and direction unit vector of one ray.
    vec3d vert2, direct2 - vertex and direction unit vector of other ray.
    vec3d res - output buffer for midpoint result.

    Returns:
    The minimal distance between the skew lines.
*/
double skew_midpoint(vec3d vert1, vec3d direct1, vec3d vert2, vec3d direct2,
    vec3d res)
{
    vec3d perp_both, sp_diff, on1, on2, temp;
    double scale;

    /* vector between starting points */
    vec_subt(vert2, vert1, sp_diff);

    /* The shortest distance is on a line perpendicular to both rays. */
    vec_cross(direct1, direct2, perp_both);
    scale = vec_dot(perp_both, perp_both);

    /* position along each ray */
    vec_cross(sp_diff, direct2, temp);
    vec_scalar_mul(direct1, vec_dot(perp_both, temp)/scale, temp);
    vec_add(vert1, temp, on1);

    vec_cross(sp_diff, direct1, temp);
    vec_scalar_mul(direct2, vec_dot(perp_both, temp)/scale, temp);
    vec_add(vert2, temp, on2);

    /* Distance: */
    scale = vec_diff_norm(on1, on2);

    /* Average */
    vec_add(on1, on2, res);
    vec_scalar_mul(res, 0.5, res);

    return scale;
}

/*  point_position() calculates an average 3D position implied by the rays
    sent toward it from cameras through the image projections of the point.

    Arguments:
    vec2d targets[] - for each camera, the 2D metric, flat, centred coordinates 
        of the identified point projection.
    int num_cams - number of cameras ( = number of elements in ``targets``).
    mm_np *multimed_pars - multimedia parameters struct for ray tracing through
        several layers.
    Calibration* cals[] - each camera's calibration object.
    vec3d res - the result average point.

    Returns:
    The ray convergence measure, an average of skew ray distance across all
    ray pairs.
*/
double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
    Calibration* cals[], vec3d res)
{
    int cam, pair; /* loop counters */
    int num_used_pairs = 0; /* averaging accumulators */
    double dtot = 0;
    vec3d point_tot = {0., 0., 0.};

    vec3d* vertices = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d* directs = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d point;

    /* Shoot rays from all cameras. */
    for (cam = 0; cam < num_cams; cam++) {
        if (targets[cam][0] != COORD_UNUSED) {
            ray_tracing(targets[cam][0], targets[cam][1], cals[cam], 
                *multimed_pars, vertices[cam], directs[cam]);
        }
    }

    /* Check intersection distance for each pair of rays and find position */
    for (cam = 0; cam < num_cams; cam++) {
        if (targets[cam][0] == COORD_UNUSED) continue;

        for (pair = cam + 1; pair < num_cams; pair++) {
            if (targets[pair][0] == COORD_UNUSED) continue;

            num_used_pairs++;
            dtot += skew_midpoint(vertices[cam], directs[cam],
                vertices[pair], directs[pair], point);
            vec_add(point_tot, point, point_tot);
        }
    }

    free(vertices);
    free(directs);

    vec_scalar_mul(point_tot, 1./num_used_pairs, res);
    return (dtot / num_used_pairs);
}

/*  weighted_dumbbell_precision() Gives a weighted sum of dumbbell precision
    measures: ray convergence, and dumbbell length. The weight of the first is
    1, and the later is weighted by a user defined value.

    Arguments:
    (vec2d **) targets - 2D array of targets, so order 3 tensor of shape
        (num_targs,num_cams,2). Each target is the 2D metric, flat, centred
        coordinates of one identified point.
    int num_targs - the number of known targets, assumed to be the same in all
        cameras.
    int num_cams - number of cameras.
    mm_np *multimed_pars - multimedia parameters struct for ray tracing through
        several layers.
    Calibration* cals[] - each camera's calibration object.
    int db_length - distance between two consecutive targets (assuming the
        targets list is a 2 by 2 list of dumbbell frames).
    double db_weight - the weight of the average length error compared to
        the average ray convergence error.
*/
double weighted_dumbbell_precision(vec2d** targets, int num_targs, int num_cams,
    mm_np *multimed_pars, Calibration* cals[], int db_length, double db_weight)
{
    int pt;
    double dtot = 0, len_err_tot = 0, dist;
    vec3d res[2], *res_current;

    for (pt = 0; pt < num_targs; pt++) {
        res_current = &(res[pt % 2]);
        dtot += point_position(targets[pt], num_cams, multimed_pars, cals,
            *res_current);

        if (pt % 2 == 1) {
            vec_subt(res[0], res[1], res[0]);
            dist = vec_norm(res[0]);
            len_err_tot += 1 - ((dist > db_length) ? (db_length/dist) : dist/db_length);
        }
    }

    /* note half as many pairs as targets is assumed */
    return (dtot / num_targs + db_weight*len_err_tot/(0.5*num_targs));
}

/*  num_deriv_exterior() calculates the partial numerical derivative of image
    coordinates of a given 3D position, over each of the 6 exterior orientation
    parameters (3 position parameters, 3 rotation angles).

    Arguments:
    Calibration* cal_in  -      camera calibration object
    control_par *cpar -     control parameters

    double dpos, double dang - the step size for numerical differentiation,
        dpos for the metric variables, dang for the angle variables. Units
        are same as the units of the variables derived.
    vec3d pos - the current 3D position represented on the image.
    
    Return parameters:
    double x_ders[6], y_ders[6] respectively the derivatives of the x and y
        image coordinates as function of each of the orientation parameters.
 */
void num_deriv_exterior(Calibration* cal, control_par *cpar, double dpos, double dang, vec3d pos,
    double x_ders[6], double y_ders[6])
{
    int pd; 
    double step, xs, ys, xpd, ypd;
    double* vars[6];
    
    vars[0] = &(cal->ext_par.x0);
    vars[1] = &(cal->ext_par.y0);
    vars[2] = &(cal->ext_par.z0);
    vars[3] = &(cal->ext_par.omega);
    vars[4] = &(cal->ext_par.phi);
    vars[5] = &(cal->ext_par.kappa);
    
    /* Starting image position */
    rotation_matrix(&(cal->ext_par));
    img_coord (pos, cal, cpar->mm, &xs, &ys);
    
    for (pd = 0; pd < 6; pd++) {
        step = (pd > 2) ? dang : dpos;
        
        *(vars[pd]) += step;
        if (pd > 2) rotation_matrix(&(cal->ext_par));
        img_coord (pos, cal, cpar->mm, &xpd, &ypd);
        x_ders[pd] = (xpd - xs) / step;
        y_ders[pd] = (ypd - ys) / step;
        *(vars[pd]) -= step;
    }
    rotation_matrix(&(cal->ext_par));
}

/*  orient() calculates orientation of the camera, updating its calibration 
    structure using the definitions and algorithms well described in [1].
    
    Arguments:
    Calibration* cal_in - camera calibration object
    control_par *cpar - control parameters
    int nfix - number of 3D known points
    vec3d fix[]	- each of nfix items is one 3D position of known point on
        the calibration object.
    target pix[] - image coordinates corresponding to each point in ``fix``.
        can be obtained from the set of detected 2D points using 
        sortgrid(). The points which are associated with fix[] have real 
        pointer (.pnr attribute), others have -999.
    orient_par flags - structure of all the flags of the parameters to be 
        (un)changed, read from orient.par parameter file using 
        read_orient_par(), defaults are zeros except for x_scale which is
        by default 1.
    
    Output:
    Calibration *cal_in - if the orientation routine converged, this structure
    is updated, otherwise, returned untouched. The routine works on a copy of
    the calibration structure, cal.
    double sigmabeta[] - array of deviations for each of the interior and 
        exterior parameters and glass interface vector (19 in total).

    Returns:
    On success, a pointer to an array of residuals. For each observation point
    i = 0..n-1, residual 2*i is the Gauss-Markof residual for the x coordinate
    and residual 2*i + 1 is for the y. Then come 10 cells with the delta 
    between initial guess and final solution for internal and distortion 
    parameters, which are also part of the G-M model and described in it.
    On failure returns NULL.
*/
double* orient (Calibration* cal_in, control_par *cpar, int nfix, vec3d fix[],
            target pix[], orient_par *flags, double sigmabeta[20]) 
{
    int  	i,j,n, itnum, stopflag, n_obs=0, maxsize;

    double  ident[IDT], XPX[NPAR][NPAR], XPy[NPAR], beta[NPAR], omega=0;
    double xp, yp, xpd, ypd, xc, yc, r, qq, p, sumP;

    int numbers;

    double al,be,ga,nGl,e1_x,e1_y,e1_z,e2_x,e2_y,e2_z,safety_x,safety_y,safety_z;
    double *P, *y, *yh, *Xbeta, *resi;
    vec3d glass_dir, tmp_vec, e1, e2;

    double (*X)[NPAR];
    double (*Xh)[NPAR];

    Calibration *cal;

    /* small perturbation for translation/rotation in meters and in radians */
    double  dm = 0.00001,  drad = 0.0000001;

    cal = malloc (sizeof (Calibration));
    memcpy(cal, cal_in, sizeof (Calibration));

    maxsize = nfix*2 + IDT;
    
    P = (double *) calloc(maxsize, sizeof(double));
    y = (double *) calloc(maxsize, sizeof(double));
    yh = (double *) calloc(maxsize, sizeof(double));
    Xbeta = (double *) calloc(maxsize, sizeof(double));
    resi = (double *) calloc(maxsize, sizeof(double));

    X = malloc(sizeof (*X) * maxsize);
    Xh = malloc(sizeof (*Xh) * maxsize);

    for(i = 0; i < maxsize; i++) {
        for(j = 0; j < NPAR; j++) {
    	      X[i][j] = 0.0;
    	      Xh[i][j] = 0.0;
        }
        y[i] = 0;
        P[i] = 1;
    }
    
    for(i = 0; i < NPAR; i++)
        sigmabeta[i] = 0.0;

    if(flags->interfflag){
        numbers = 18;
    } else{
        numbers = 16;
    }

    vec_set(glass_dir, 
        cal->glass_par.vec_x, cal->glass_par.vec_y, cal->glass_par.vec_z);
    nGl = vec_norm(glass_dir);

    e1_x = 2*cal->glass_par.vec_z - 3*cal->glass_par.vec_x;
    e1_y = 3*cal->glass_par.vec_x - 1*cal->glass_par.vec_z;
    e1_z = 1*cal->glass_par.vec_y - 2*cal->glass_par.vec_y;
    vec_set(tmp_vec, e1_x, e1_y, e1_z);
    unit_vector(tmp_vec, e1);

    e2_x = e1_y*cal->glass_par.vec_z - e1_z*cal->glass_par.vec_x;
    e2_y = e1_z*cal->glass_par.vec_x - e1_x*cal->glass_par.vec_z;
    e2_z = e1_x*cal->glass_par.vec_y - e1_y*cal->glass_par.vec_y;
    vec_set(tmp_vec, e2_x, e2_y, e2_z);
    unit_vector(tmp_vec, e2);

    al = 0;
    be = 0;
    ga = 0;

    /* init identities */
    ident[0] = cal->int_par.cc;
    ident[1] = cal->int_par.xh;
    ident[2] = cal->int_par.yh;
    ident[3] = cal->added_par.k1;
    ident[4] = cal->added_par.k2;
    ident[5] = cal->added_par.k3;
    ident[6] = cal->added_par.p1;
    ident[7] = cal->added_par.p2;
    ident[8] = cal->added_par.scx;
    ident[9] = cal->added_par.she;

    safety_x = cal->glass_par.vec_x;
    safety_y = cal->glass_par.vec_y;
    safety_z = cal->glass_par.vec_z;
    
    /* main loop, program runs through it, until none of the beta values
      comes over a threshold and no more points are thrown out
      because of their residuals */

    itnum = 0;  
    stopflag = 0;
    while ((stopflag == 0) && (itnum < NUM_ITER)) {
      itnum++;

      for (i = 0, n = 0; i < nfix; i++) {
        /* check for correct correspondence
        note that we do not use anymore pointer in fix, the points are read by
        the order of appearance and if we want to use every other point
        we use 'i', just check it is not -999 */
        if(pix[i].pnr != i) continue;
        
        switch (flags->useflag) {
            case 1: if ((i % 2) == 0)  continue;  break;
            case 2: if ((i % 2) != 0)  continue;  break;
            case 3: if ((i % 3) == 0)  continue;  break;
        }

        /* get metric flat-image coordinates of the detected point */
        pixel_to_metric (&xc, &yc, pix[i].x, pix[i].y, cpar);
        correct_brown_affin (xc, yc, cal->added_par, &xc, &yc);

        /* Projected 2D position on sensor of corresponding known point */
        rotation_matrix(&(cal->ext_par));
        img_coord (fix[i], cal, cpar->mm, &xp, &yp);

        /* derivatives of distortion parameters */

        r = sqrt (xp*xp + yp*yp);

        X[n][7] = cal->added_par.scx;
        X[n+1][7] = sin(cal->added_par.she);

        X[n][8] = 0;
        X[n+1][8] = 1;

        X[n][9] = cal->added_par.scx * xp * r*r;
        X[n+1][9] = yp * r*r;

        X[n][10] = cal->added_par.scx * xp * pow(r,4.0);
        X[n+1][10] = yp * pow(r,4.0);

        X[n][11] = cal->added_par.scx * xp * pow(r,6.0);
        X[n+1][11] = yp * pow(r,6.0);

        X[n][12] = cal->added_par.scx * (2*xp*xp + r*r);
        X[n+1][12] = 2 * xp * yp;

        X[n][13] = 2 * cal->added_par.scx * xp * yp;
        X[n+1][13] = 2*yp*yp + r*r;

        qq =  cal->added_par.k1*r*r; qq += cal->added_par.k2*pow(r,4.0);
        qq += cal->added_par.k3*pow(r,6.0);
        qq += 1;
        X[n][14] = xp * qq + cal->added_par.p1 * (r*r + 2*xp*xp) + \
                                                            2*cal->added_par.p2*xp*yp;
        X[n+1][14] = 0;

        X[n][15] = -cos(cal->added_par.she) * yp;
        X[n+1][15] = -sin(cal->added_par.she) * yp;

        /* numeric derivatives of projection coordinates over external 
           parameters, 3D position and the angles */
        
        num_deriv_exterior(cal, cpar, dm, drad, fix[i], X[n], X[n + 1]);

        /* Num. deriv. of projection coords over sensor distance from PP */
        cal->int_par.cc += dm;
        rotation_matrix(&(cal->ext_par));
        img_coord (fix[i], cal, cpar->mm, &xpd, &ypd);
        X[n][6]   = (xpd - xp) / dm;
        X[n+1][6] = (ypd - yp) / dm;
        cal->int_par.cc -= dm;

        /* ditto, over water-glass-air interface position vector */
        al += dm;
        cal->glass_par.vec_x += e1[0]*nGl*al;
        cal->glass_par.vec_y += e1[1]*nGl*al;
        cal->glass_par.vec_z += e1[2]*nGl*al;

        img_coord (fix[i], cal, cpar->mm, &xpd, &ypd);
        X[n][16]      = (xpd - xp) / dm;
        X[n+1][16] = (ypd - yp) / dm;

        al -= dm;
        cal->glass_par.vec_x = safety_x;
        cal->glass_par.vec_y = safety_y;
        cal->glass_par.vec_z = safety_z;

        be += dm;
        cal->glass_par.vec_x += e2[0]*nGl*be;
        cal->glass_par.vec_y += e2[1]*nGl*be;
        cal->glass_par.vec_z += e2[2]*nGl*be;

        img_coord (fix[i], cal, cpar->mm, &xpd, &ypd);
        X[n][17]      = (xpd - xp) / dm;
        X[n+1][17] = (ypd - yp) / dm;

        be -= dm;
        cal->glass_par.vec_x = safety_x;
        cal->glass_par.vec_y = safety_y;
        cal->glass_par.vec_z = safety_z;

        ga += dm;
        cal->glass_par.vec_x += cal->glass_par.vec_x*nGl*ga;
        cal->glass_par.vec_y += cal->glass_par.vec_y*nGl*ga;
        cal->glass_par.vec_z += cal->glass_par.vec_z*nGl*ga;

        img_coord (fix[i], cal, cpar->mm, &xpd, &ypd);
        X[n][18]      = (xpd - xp) / dm;
        X[n+1][18] = (ypd - yp) / dm;

        ga -= dm;
        cal->glass_par.vec_x = safety_x;
        cal->glass_par.vec_y = safety_y;
        cal->glass_par.vec_z = safety_z;

        y[n]   = xc - xp;
        y[n+1] = yc - yp;

        n += 2;
      }
      
      n_obs = n;
      
      /* identities */
      for (i = 0; i < IDT; i++)
        X[n_obs + i][6 + i] = 1;
        
      y[n_obs+0] = ident[0] - cal->int_par.cc;
      y[n_obs+1] = ident[1] - cal->int_par.xh;
      y[n_obs+2] = ident[2] - cal->int_par.yh;
      y[n_obs+3] = ident[3] - cal->added_par.k1;
      y[n_obs+4] = ident[4] - cal->added_par.k2;
      y[n_obs+5] = ident[5] - cal->added_par.k3;
      y[n_obs+6] = ident[6] - cal->added_par.p1;
      y[n_obs+7] = ident[7] - cal->added_par.p2;
      y[n_obs+8] = ident[8] - cal->added_par.scx;
      y[n_obs+9] = ident[9] - cal->added_par.she;

      /* weights */
      for (i = 0; i < n_obs; i++)
          P[i] = 1;

      P[n_obs+0] = ( ! flags->ccflag) ?  POS_INF : 1;
      P[n_obs+1] = ( ! flags->xhflag) ?  POS_INF : 1;
      P[n_obs+2] = ( ! flags->yhflag) ?  POS_INF : 1;
      P[n_obs+3] = ( ! flags->k1flag) ?  POS_INF : 1;
      P[n_obs+4] = ( ! flags->k2flag) ?  POS_INF : 1;
      P[n_obs+5] = ( ! flags->k3flag) ?  POS_INF : 1;
      P[n_obs+6] = ( ! flags->p1flag) ?  POS_INF : 1;
      P[n_obs+7] = ( ! flags->p2flag) ?  POS_INF : 1;
      P[n_obs+8] = ( ! flags->scxflag) ?  POS_INF : 1;
      P[n_obs+9] = ( ! flags->sheflag) ?  POS_INF : 1;

      n_obs += IDT;
      sumP = 0;
      for (i = 0; i < n_obs; i++) {       	/* homogenize */
          p = sqrt (P[i]);
          for (j = 0; j < NPAR; j++)
              Xh[i][j] = p * X[i][j];
            
          yh[i] = p * y[i];
          sumP += P[i];
      }
        
      /* Gauss Markoff Model it is the least square adjustment 
         of the redundant information contained both in the spatial 
         intersection and the resection, see [1], eq. 23 */
      ata ((double *) Xh, (double *) XPX, n_obs, numbers, NPAR );
      matinv ((double *) XPX, numbers, NPAR);
      atl ((double *) XPy, (double *) Xh, yh, n_obs, numbers, NPAR);
      matmul ((double *) beta, (double *) XPX, (double *) XPy, 
          numbers, numbers,1, NPAR, NPAR);

      stopflag = 1;
      for (i = 0; i < numbers; i++) {
          if (fabs (beta[i]) > CONVERGENCE)  stopflag = 0;
      }

      if ( ! flags->ccflag) beta[6] = 0.0;
      if ( ! flags->xhflag) beta[7] = 0.0;
      if ( ! flags->yhflag) beta[8] = 0.0;
      if ( ! flags->k1flag) beta[9] = 0.0;
      if ( ! flags->k2flag) beta[10] = 0.0;
      if ( ! flags->k3flag) beta[11] = 0.0;
      if ( ! flags->p1flag) beta[12] = 0.0;
      if ( ! flags->p2flag) beta[13] = 0.0;
      if ( ! flags->scxflag)beta[14] = 0.0;
      if ( ! flags->sheflag) beta[15] = 0.0;

      cal->ext_par.x0 += beta[0];
      cal->ext_par.y0 += beta[1];
      cal->ext_par.z0 += beta[2];
      cal->ext_par.omega += beta[3];
      cal->ext_par.phi += beta[4];
      cal->ext_par.kappa += beta[5];
      cal->int_par.cc += beta[6];
      cal->int_par.xh += beta[7];
      cal->int_par.yh += beta[8];
      cal->added_par.k1 += beta[9];
      cal->added_par.k2 += beta[10];
      cal->added_par.k3 += beta[11];
      cal->added_par.p1 += beta[12];
      cal->added_par.p2 += beta[13];
      cal->added_par.scx += beta[14];
      cal->added_par.she += beta[15];

      if (flags->interfflag) {
          cal->glass_par.vec_x += e1[0]*nGl*beta[16];
          cal->glass_par.vec_y += e1[1]*nGl*beta[16];
          cal->glass_par.vec_z += e1[2]*nGl*beta[16];
          cal->glass_par.vec_x += e2[0]*nGl*beta[17];
          cal->glass_par.vec_y += e2[1]*nGl*beta[17];
          cal->glass_par.vec_z += e2[2]*nGl*beta[17];
      }
    }

    /* compute residuals etc. */
    matmul ( (double *) Xbeta, (double *) X, (double *) beta, n_obs, 
        numbers, 1, n_obs, NPAR);
    omega = 0;
    for (i = 0; i < n_obs; i++) {
        resi[i] = Xbeta[i] - y[i];
        omega += resi[i] * P[i] * resi[i];
    }
    sigmabeta[NPAR] = sqrt (omega / (n_obs - numbers));

    for (i = 0; i < numbers; i++) { 
        sigmabeta[i] = sigmabeta[NPAR] * sqrt(XPX[i][i]);
    }

    free(X);
    free(P);
    free(y);
    free(Xbeta);
    free(Xh);

    if (stopflag){
        rotation_matrix(&(cal->ext_par));
        memcpy(cal_in, cal, sizeof (Calibration));
        return resi;
    }
    else {
        free(resi);
        return NULL;
    }
}

/*  raw_orient() uses manually clicked points to setup the first raw orientation of the
                camera, setting its external parameters: position and angles.
    Calibration* cal  -      camera calibration object
    control_par *cpar -     control parameters
    int nfix	-	number of points, for raw_orient this means typically 4 points
                    manually selected by the user
    vec3d fix[]	-	object point data obtained using read_calblock() function and
                    represents the known calibration body 3D points for the 4 manually
                    entered points, see man_ori.par
    target pix[] -	image coordinates obtained from sortgrid(). The points which are
                    associated with fix[] have real pointer (i+1), others have -999,
                    for raw_orient these will be typically pix0[] - manually clicked dots
    Output:
    Calibration *cal overwritten with the updated orientation, only 6 external parameters
                     being updated: x,y,z,omega,phi,kappa
    returns 1 for success, 0 for failure of convergence

*/


int raw_orient (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], target pix[])
{
    double  X[10][6], y[10], XPX[6][6], XPy[6], beta[6];
    int     i, j, n, itnum, stopflag;
    double  dm = 0.0001,  drad = 0.0001;
    double 	xp, yp, xc, yc;
    vec3d   pos;

    /* init X, y (set to zero) */
    for (i = 0; i < 10; i++) {
      for (j = 0; j < 6; j++)
        X[i][j] = 0;
      y[i] = 0;
    }

    cal->added_par.k1 = 0;
    cal->added_par.k2 = 0;
    cal->added_par.k3 = 0;
    cal->added_par.p1 = 0;
    cal->added_par.p2 = 0;
    cal->added_par.scx = 1;
    cal->added_par.she = 0;

    /* main loop, program runs through it, until none of the beta values
     comes over a threshold and no more points are thrown out
     because of their residuals */

    itnum = 0;
    stopflag = 0;

    while ((stopflag == 0) && (itnum < 20)) {
        ++itnum;

        for (i = 0, n = 0; i < nfix; i++) {
            /* we do not check the order - trust the user to click the points
               in the correct order of appearance in man_ori and in the calibration
               parameters GUI
            */
            pixel_to_metric (&xc, &yc, pix[i].x, pix[i].y, cpar);
            /* no corrections as additional parameters are neglected
                correct_brown_affin (xc, yc, cal->added_par, &xc, &yc);
            */
    
            /* every calibration dot is projected to the mm position, xp, yp */
            vec_set(pos, fix[i][0], fix[i][1], fix[i][2]);
            rotation_matrix(&(cal->ext_par));
            img_coord (pos, cal, cpar->mm, &xp, &yp);
    
            /* numeric derivatives of internal camera coefficients */
            num_deriv_exterior(cal, cpar, dm, drad, pos, X[n], X[n + 1]);
    
            y[n]   = xc - xp;
            y[n+1] = yc - yp;
    
            n += 2;
        }

        /* Gauss Markoff Model */
    
        ata ((double *) X, (double *) XPX, n, 6, 6);
        matinv ((double *) XPX, 6, 6);
        atl ((double *) XPy, (double *) X, y, n, 6, 6);
        matmul ((double *) beta, (double *) XPX, (double *) XPy, 6,6,1,6,6);
    
        stopflag = 1;
        for (i = 0; i < 6; i++) {
          if (fabs (beta[i]) > 0.1 )
            stopflag = 0;
        }
    
        cal->ext_par.x0 += beta[0];
        cal->ext_par.y0 += beta[1];
        cal->ext_par.z0 += beta[2];
        cal->ext_par.omega += beta[3];
        cal->ext_par.phi += beta[4];
        cal->ext_par.kappa += beta[5];
    
    }

    if (stopflag) {
        rotation_matrix(&(cal->ext_par));
    }
    return stopflag;
}

/* read_man_ori_fix() reads the locations of the points selected for the manual
    orientation

   Arguments:
   vec3d fix4[] structure 3d positions and integer identification pointers of
   the calibration target points in the calibration file
   char *calblock_filename - path to the text file containing the calibration points.
   char* man_ori_filename - path to the text file containing the manually selected points.
   int cam - ID (number) of the camera (0,1, ..., n_cams)

   Returns:
   int number of points, should be 4. If reading failed for any reason, returns NULL.
*/
int read_man_ori_fix(vec3d fix4[4], char* calblock_filename, 
    char* man_ori_filename, int cam)
{
    FILE* fpp;
    int	dummy, pnr, nr[4], i;
    int num_fix, num_match;
    vec3d *fix = NULL;
    int ret;

    fpp = fopen(man_ori_filename, "r");
    if (!fpp) {
        printf("Can't open manual orientation file %s\n", man_ori_filename);
        goto handle_error;
    }
    printf("Opened %s\n", man_ori_filename);

    /* Skip cameras before the one we're interested in */
    for (i = 0; i < cam; i++) {
        ret = fscanf(fpp, "%d %d %d %d \n", &dummy, &dummy, &dummy, &dummy);
        if (ret != 4) {
            printf("Error reading dummy values for camera %d\n", i);
            goto handle_error;
        }
    }
    printf("Skipped %d cameras\n", cam);
    
    /* Read the point numbers for our camera */
    ret = fscanf(fpp, "%d %d %d %d \n", &nr[0], &nr[1], &nr[2], &nr[3]);
    if (ret != 4) {
        printf("Error reading point numbers\n");
        goto handle_error;
    }
    printf("Looking for points: %d %d %d %d\n", nr[0], nr[1], nr[2], nr[3]);
    
    fclose(fpp);

    /* read the id and positions of the fixed points, assign the pre-defined to fix4 */
    fix = read_calblock(&num_fix, calblock_filename);
    if (num_fix < 4) {
        printf("Too few points or incompatible file: %s (got %d points)\n", 
            calblock_filename, num_fix);
        goto handle_error;
    }
    printf("Read %d points from calblock\n", num_fix);
    
    num_match = 0; /* count matches to needed numbers */
    for (pnr = 0; pnr < num_fix; pnr++) {
        printf("Checking point %d\n", pnr);
        for (i = 0; i < 4; i++) {
            if (pnr == nr[i] - 1) {
                vec_copy(fix4[i], fix[pnr]);
                num_match++;
                printf("Found match %d: point %d matches needed point %d\n", 
                    num_match, pnr, nr[i]);
                break;
            }
        }
        if (num_match >= 4) break;
    }
    
    free(fix);   
    printf("Found %d matches total\n", num_match);
    return num_match;

handle_error:
    if (fix) free(fix);
    return 0;
}


/* Reads orientation parameters from file.
 * Parameter: filename - the absolute/relative path to file to be read.
 * Returns: pointer to a new orient_par structure.
 */
orient_par* read_orient_par(char *filename) {
    orient_par *ret;
    FILE * file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open orientation parameters file %s.\n", filename);
        return NULL;
    }

    ret = malloc(sizeof(orient_par));

    if (   !(fscanf(file, "%d", &ret->useflag)==1)  /* use every point or every other pt */
        || !(fscanf(file, "%d", &ret->ccflag)==1)   /* change back focal distance */
        || !(fscanf(file, "%d", &ret->xhflag)==1)   /* change xh point, 1-yes, 0-no */
        || !(fscanf(file, "%d", &ret->yhflag)==1)   /* change yh point */
        || !(fscanf(file, "%d", &ret->k1flag)==1)   /* change k1 */
        || !(fscanf(file, "%d", &ret->k2flag)==1)   /* change k2 */
        || !(fscanf(file, "%d", &ret->k3flag)==1)   /* k3  */
        || !(fscanf(file, "%d", &ret->p1flag)==1)    /* p1  */
        || !(fscanf(file, "%d", &ret->p2flag)==1)     /* p2 */
        || !(fscanf(file, "%d", &ret->scxflag)==1)   /* scx - scaling  */
        || !(fscanf(file, "%d", &ret->sheflag)==1)    /* she - shearing  */
        || !(fscanf(file, "%d", &ret->interfflag)==1))  /* interface glass vector */
    {
        printf("Error reading orientation parameters from %s\n", filename);
        free(ret);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return ret;
}
