
#include "orientation.h"
#include "calibration.h"
#include "ray_tracing.h"
#include "vec_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PT_UNUSED -999
#define NUM_ITER  80
#define POS_INF 1e20

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
    vec2d targets[] - for each camera, the 2D metric coordinates of the 
        identified point projection.
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
    
    vec2d current;
    vec3d* vertices = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d* directs = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d point;
    
    /* Shoot rays from all cameras. */
    for (cam = 0; cam < num_cams; cam++) {
        if (targets[cam][0] != PT_UNUSED) {
            current[0] = targets[cam][0] - cals[cam]->int_par.xh;
            current[1] = targets[cam][1] - cals[cam]->int_par.yh;
            
            ray_tracing(current[0], current[1], cals[cam], *multimed_pars,
                vertices[cam], directs[cam]);
        }
    }
    
    /* Check intersection distance for each pair of rays and find position */
    for (cam = 0; cam < num_cams; cam++) {
        if (targets[cam][0] == PT_UNUSED) continue;
        
        for (pair = cam + 1; pair < num_cams; pair++) {
            if (targets[pair][0] == PT_UNUSED) continue;
            
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
        (num_targs,num_cams,2). Each target is the 2D metric coordinates of 
        one identified point.
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

/*
    mm_np *mm       - multimedia parameters struct for ray tracing through several layers.
    Calibration* cal  -      camera calibration object.
    int nfix	-	 # of object points 
    vec3d fix[]	-	 object point data 
    vec2d crd[]	-	 image coordinates 
    int ncam  -		image number for residual display 
*/


void orient_v3 (Calibration* cal, mm_np *mm, int nfix, vec3d fix[], vec2d crd[], int ncam)
{
    int  	i,j,n, itnum, stopflag, n_obs=0, convergeflag;
    int  	useflag, ccflag, scxflag, sheflag, interfflag, xhflag, yhflag,
            k1flag, k2flag, k3flag, p1flag, p2flag;
    int  	intx1, intx2, inty1, inty2;
    
    double  ident[10], XPX[19][19], XPy[19], beta[19], omega=0, sigma0, sigmabeta[19]; 
    double 	xp, yp, xpd, ypd, r, qq, p, sumP;
    
    int dummy, multi,numbers;
    
    FILE *par_file;
    double al,be,ga,nGl,e1_x,e1_y,e1_z,e2_x,e2_y,e2_z,n1,n2,safety_x,safety_y,safety_z;
    double *P, *y, *Xbeta, *resi, *pixnr;
    vec3d glass_dir, tmp_vec, e1, e2, pos;
    
    /* delta in meters and in radians for derivatives */
    double  dm = 0.00001,  drad = 0.0000001; 
    
    /* we need to preserve the calibration if the model does not converge */
    Calibration *safety_cal;
    memcpy(&safety_cal, &cal, sizeof (Calibration*));
    
    
    /* memory allocation */
    P = (double *) calloc(nfix, sizeof(double));
    y = (double *) calloc(nfix, sizeof(double));
    Xbeta = (double *) calloc(nfix, sizeof(double));
    resi = (double *) calloc(nfix, sizeof(double));
    pixnr = (double *) calloc(nfix*2, sizeof(double));
    
    double **X = malloc(sizeof (*X) * nfix);
    double **Xh = malloc(sizeof (*Xh) * nfix);
    if (X != NULL)
    {
      for (i = 0; i < nfix; i++)
      {
        X[i] = malloc(19*sizeof(double));
        Xh[i] = malloc(19*sizeof(double));
      }
    }
    else {
        printf(" Memory allocation failed \n");
    }        
    
    /* fill with zeros */
	for(i = 0; i < nfix; i++)
		{
		for(j = 0; j < 19; j++)
			X[i][j] = 0;
		}


    /* read, which parameters shall be used */
    par_file = fopen("parameters/orient.par","r");
    if (par_file != NULL){     
          fscanf (par_file,"%d", &useflag);
          fscanf (par_file,"%d", &ccflag);
          fscanf (par_file,"%d", &xhflag);
          fscanf (par_file,"%d", &yhflag);
          fscanf (par_file,"%d", &k1flag);
          fscanf (par_file,"%d", &k2flag);
          fscanf (par_file,"%d", &k3flag);
          fscanf (par_file,"%d", &p1flag);
          fscanf (par_file,"%d", &p2flag);
          fscanf (par_file,"%d", &scxflag);
          fscanf (par_file,"%d", &sheflag);
          fscanf (par_file,"%d", &interfflag);
          fclose (par_file);
    }
    else
    { 
        printf("Failed to read orient.par\n");
    }


    vec_set(glass_dir, cal->glass_par.vec_x, cal->glass_par.vec_y, cal->glass_par.vec_z);
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
    ident[0] = cal->int_par.cc;  ident[1] = cal->int_par.xh; ident[2] = cal->int_par.yh;
    ident[3] = cal->added_par.k1; ident[4]=cal->added_par.k2;ident[5] = cal->added_par.k3;
    ident[6] = cal->added_par.p1; ident[7] = cal->added_par.p2;
    ident[8] = cal->added_par.scx; ident[9] = cal->added_par.she;

    /* main loop, program runs through it, until none of the beta values
     comes over a threshold and no more points are thrown out
     because of their residuals */


    safety_x = cal->glass_par.vec_x;
    safety_y = cal->glass_par.vec_y;
    safety_z = cal->glass_par.vec_z;

    itnum = 0;  stopflag = 0;
    while ((stopflag == 0) && (itnum < NUM_ITER))
        {
  
          itnum++;
    
          for (i=0, n=0; i<nfix; i++) if (crd[i].pnr == fix[i].pnr) 
          {
          /* use only certain points as control points */
          switch (useflag)
            {
            case 1: if ((fix[i].pnr % 2) == 0)  continue;  break;
            case 2: if ((fix[i].pnr % 2) != 0)  continue;  break;
            case 3: if ((fix[i].pnr % 3) == 0)  continue;  break;
            }

          /* check for correct correspondence */
          if (crd[i].pnr != fix[i].pnr)	continue;


          pixnr[n/2] = i;		/* for drawing residuals */
          vec_set(pos, fix[i].x, fix[i].y, fix[i].z);
          rotation_matrix(&(cal->ext_par));
          img_coord (pos, cal, mm, &xp, &yp);
  
      /*
          if ((i % 100) == 0){
           printf("\n %d %f %f %f %d %d %f %f \n",i,Xp,Yp,Zp,crd[i].pnr, fix[i].pnr,xp,yp);
          }
     */
      
          /* derivatives of additional parameters */

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
          X[n][14] = xp * qq + cal->added_par.p1 * (r*r + 2*xp*xp) + 2*cal->added_par.p2*xp*yp;
          X[n+1][14] = 0;

          X[n][15] = -cos(cal->added_par.she) * yp;
          X[n+1][15] = -sin(cal->added_par.she) * yp;



          /* numeric derivatives */
          
          cal->ext_par.x0 += dm;
          img_coord (pos, cal, mm, &xpd, &ypd); 
          X[n][0]      = (xpd - xp) / dm;
          X[n+1][0] = (ypd - yp) / dm;
          cal->ext_par.x0 -= dm;

          cal->ext_par.y0 += dm;
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][1]   = (xpd - xp) / dm;
          X[n+1][1] = (ypd - yp) / dm;
          cal->ext_par.y0 -= dm;

          cal->ext_par.z0 += dm;
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][2]      = (xpd - xp) / dm;
          X[n+1][2] = (ypd - yp) / dm;
          cal->ext_par.z0 -= dm;

          cal->ext_par.omega += drad;
          rotation_matrix(&(cal->ext_par));
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][3]      = (xpd - xp) / drad;
          X[n+1][3] = (ypd - yp) / drad;
          cal->ext_par.omega -= drad;

          cal->ext_par.phi += drad;
          rotation_matrix(&(cal->ext_par));
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][4]      = (xpd - xp) / drad;
          X[n+1][4] = (ypd - yp) / drad;
          cal->ext_par.phi -= drad;

          cal->ext_par.kappa += drad;
          rotation_matrix(&(cal->ext_par));
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][5]      = (xpd - xp) / drad;
          X[n+1][5] = (ypd - yp) / drad;
          cal->ext_par.kappa -= drad;

          cal->int_par.cc += dm;
          rotation_matrix(&(cal->ext_par));
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][6]      = (xpd - xp) / dm;
          X[n+1][6] = (ypd - yp) / dm;
          cal->int_par.cc -= dm;
  

          al += dm;
          cal->glass_par.vec_x += e1[0]*nGl*al;
          cal->glass_par.vec_y += e1[1]*nGl*al;
          cal->glass_par.vec_z += e1[2]*nGl*al;
      
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][16]      = (xpd - xp) / dm;
          X[n+1][16] = (ypd - yp) / dm;
 
          al -= dm;
          cal->glass_par.vec_x = safety_x;
          cal->glass_par.vec_y = safety_y;
          cal->glass_par.vec_z = safety_z;

          //cal->glass_par.vec_y += dm;
          be += dm;
          cal->glass_par.vec_x += e2[0]*nGl*be;
          cal->glass_par.vec_y += e2[1]*nGl*be;
          cal->glass_par.vec_z += e2[2]*nGl*be;
      
          img_coord (pos, cal, mm, &xpd, &ypd);
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
      
          img_coord (pos, cal, mm, &xpd, &ypd);
          X[n][18]      = (xpd - xp) / dm;
          X[n+1][18] = (ypd - yp) / dm;

          ga -= dm;
          cal->glass_par.vec_x = safety_x;
          cal->glass_par.vec_y = safety_y;
          cal->glass_par.vec_z = safety_z;
  
          y[n]   = crd[i].x - xp;
          y[n+1] = crd[i].y - yp;

          n += 2;
        } // end if crd == fix
  
  
          n_obs = n;
  
          /* identities */

          for (i=0; i<10; i++)  X[n_obs+i][6+i] = 1;

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
          for (i=0; i<n_obs; i++)  P[i] = 1;
      
          if ( ! ccflag)  P[n_obs+0] = POS_INF;
          if ( ! xhflag)  P[n_obs+1] = POS_INF;
          if ( ! yhflag)  P[n_obs+2] = POS_INF;
          if ( ! k1flag)  P[n_obs+3] = POS_INF;
          if ( ! k2flag)  P[n_obs+4] = POS_INF;
          if ( ! k3flag)  P[n_obs+5] = POS_INF;
          if ( ! p1flag)  P[n_obs+6] = POS_INF;
          if ( ! p2flag)  P[n_obs+7] = POS_INF;
          if ( ! scxflag) P[n_obs+8] = POS_INF;
          if ( ! sheflag) P[n_obs+9] = POS_INF;


          n_obs += 10;  sumP = 0;
          for (i=0; i<n_obs; i++)	       	/* homogenize */
        {
          p = sqrt (P[i]);
          for (j=0; j<19; j++)  Xh[i][j] = p * X[i][j];
          yh[i] = p * y[i];  sumP += P[i];
        }



          /* Gauss Markoff Model */
          numbers = 16;
          
          if(interfflag){
             numbers = 18;
          }
  
          ata ((double *) Xh, (double *) XPX, n_obs, numbers, 19 );
          matinv ((double *) XPX, numbers, 19);
          atl ((double *) XPy, (double *) Xh, yh, n_obs, numbers, 19);
          matmul ((double *) beta, (double *) XPX, (double *) XPy, numbers,numbers,1,19,19);
  
          stopflag = 1;
          convergeflag = 1;
          for (i=0; i<numbers; i++)
        {
          if (fabs (beta[i]) > 0.0001)  stopflag = 0;	/* more iterations */////Achtung
          if (fabs (beta[i]) > 0.01)  convergeflag = 0;
        }
  
          if ( ! ccflag) beta[6] = 0.0;
          if ( ! xhflag) beta[7] = 0.0;
          if ( ! yhflag) beta[8] = 0.0;
          if ( ! k1flag) beta[9] = 0.0;
          if ( ! k2flag) beta[10] = 0.0;
          if ( ! k3flag) beta[11] = 0.0;
          if ( ! p1flag) beta[12] = 0.0;
          if ( ! p2flag) beta[13] = 0.0;
          if ( ! scxflag)beta[14] = 0.0;
          if ( ! sheflag) beta[15] = 0.0;
  
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
      
          if(interfflag)
          {
              cal->glass_par.vec_x += e1[0]*nGl*beta[16];
              cal->glass_par.vec_y += e1[1]*nGl*beta[16];
              cal->glass_par.vec_z += e1[2]*nGl*beta[16];
              cal->glass_par.vec_x += e2[0]*nGl*beta[17]; 
              cal->glass_par.vec_y += e2[1]*nGl*beta[17];
              cal->glass_par.vec_z += e2[2]*nGl*beta[17];
          }
          beta[0]=beta[0];
    } // end of while iterations and stopflag



    /* compute residuals etc. */

    matmul ( (double *) Xbeta, (double *) X, (double *) beta, n_obs, numbers, 1, n_obs, 19);
    omega = 0;
    for (i=0; i<n_obs; i++)
    {
      resi[i] = Xbeta[i] - y[i];  
      omega += resi[i] * P[i] * resi[i];
    }
    sigma0 = sqrt (omega / (n_obs - numbers));

    for (i=0; i<numbers; i++)  sigmabeta[i] = sigma0 * sqrt(XPX[i][i]);


    /* correlations between parameters */
    /*if (examine)	for (i=0; i<18; i++)
    {
      for (j=0; j<18; j++)
    printf ("%6.2f",
        XPX[i][j] / (sqrt(XPX[i][i]) * sqrt(XPX[j][j])));
      printf ("\n");
    }*/


    /* print results */
    printf ("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
    printf ("\n\nResults after %d iterations:\n\n", itnum);
    printf ("sigma0 = %6.2f micron\n", sigma0*1000);
    printf ("X0 =    %8.3f   +/- %8.3f\n", cal->ext_par.x0, sigmabeta[0]);
    printf ("Y0 =    %8.3f   +/- %8.3f\n", cal->ext_par.y0, sigmabeta[1]);
    printf ("Z0 =    %8.3f   +/- %8.3f\n", cal->ext_par.z0, sigmabeta[2]);
    printf ("omega = %8.4f   +/- %8.4f\n", cal->ext_par.omega*ro, sigmabeta[3]*ro);
    printf ("phi   = %8.4f   +/- %8.4f\n", cal->ext_par.phi*ro, sigmabeta[4]*ro);
    printf ("kappa = %8.4f   +/- %8.4f\n", cal->ext_par.kappa*ro, sigmabeta[5]*ro);
    if(interfflag){
    printf ("cal->glass_par.vec_x = %8.4f   +/- %8.4f\n", cal->glass_par.vec_x/nGl, \
                                                        (sigmabeta[16]+sigmabeta[17]));
    printf ("cal->glass_par.vec_y = %8.4f   +/- %8.4f\n", cal->glass_par.vec_y/nGl, \
                                                        (sigmabeta[16]+sigmabeta[17]));
    printf ("cal->glass_par.vec_z = %8.4f   +/- %8.4f\n", cal->glass_par.vec_z/nGl, \
                                                        (sigmabeta[16]+sigmabeta[17]));
    }
    //printf ("vec_z = %8.4f   +/- %8.4f\n", cal->glass_par.vec_z, sigmabeta[18]);
    printf ("camera const  = %8.5f   +/- %8.5f\n", cal->int_par.cc, sigmabeta[6]);
    printf ("xh            = %8.5f   +/- %8.5f\n", cal->int_par.xh, sigmabeta[7]);
    printf ("yh            = %8.5f   +/- %8.5f\n", cal->int_par.yh, sigmabeta[8]);
    printf ("k1            = %8.5f   +/- %8.5f\n", cal->added_par.k1, sigmabeta[9]);
    printf ("k2            = %8.5f   +/- %8.5f\n", cal->added_par.k2, sigmabeta[10]);
    printf ("k3            = %8.5f   +/- %8.5f\n", cal->added_par.k3, sigmabeta[11]);
    printf ("p1            = %8.5f   +/- %8.5f\n", cal->added_par.p1, sigmabeta[12]);
    printf ("p2            = %8.5f   +/- %8.5f\n", cal->added_par.p2, sigmabeta[13]);
    printf ("scale for x'  = %8.5f   +/- %8.5f\n", cal->added_par.scx, sigmabeta[14]);
    printf ("shearing      = %8.5f   +/- %8.5f\n", cal->added_par.she*ro, \
                                                                    sigmabeta[15]*ro);


    par_file = fopen_r ("parameters/examine.par");
    fscanf (par_file,"%d\n", &dummy);
    fscanf (par_file,"%d\n", &multi);
    fclose (par_file);
    if (dummy==1){
      examine=4;
    }
    else{
      examine=0;
    }


    /* show original images with residual vectors (requires globals) */
    printf ("%d: %5.2f micron, ", nr+1, sigma0*1000);
    printf("\ntest 1 inside orientation\n");
    for (i=0; i<n_obs-10; i+=2)
    {
      n = pixnr[i/2];
      intx1 = (int) pix[nr][n].x;
      inty1 = (int) pix[nr][n].y;
      intx2 = intx1 + resi[i]*5000;  // arbitrary elongation factor for plot
      inty2 = inty1 + resi[i+1]*5000;
    orient_x1[nr][n]=intx1;
    orient_y1[nr][n]=inty1;
    orient_x2[nr][n]=intx2;
    orient_y2[nr][n]=inty2;
    orient_n[nr]=n;
    }



    if (convergeflag){
      rotation_matrix(&(cal->ext_par));
    }
    else{
        /* restore the saved calibration if not converged */	
        memcpy(&cal, &safety_cal, sizeof Calibration );
        printf ("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
        printf (" Orientation does not converge");
        printf ("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
    }
   
    for(i = 0; i < nfix; i++) {
        free(X[i]);
        free(Xh[i]);
    }
    
    free(X);
    free(P);
    free(y);
    free(Xbeta);
    free(Xh);
    free(resi);
    
}

