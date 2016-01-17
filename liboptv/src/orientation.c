
#include "orientation.h"
#include "calibration.h"
#include "ray_tracing.h"
#include "vec_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PT_UNUSED -999

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
  int  	i,j,n, itnum, stopflag, n_obs=0,convergeflag;
  int  	useflag, ccflag, scxflag, sheflag, interfflag, xhflag, yhflag,
    k1flag, k2flag, k3flag, p1flag, p2flag;
  int  	intx1, intx2, inty1, inty2;
  double       	dm = 0.00001,  drad = 0.0000001,drad2 = 0.00001, dg=0.1;
  double       	X[10000][19], Xh[10000][19], y[10000], yh[10000], ident[10],
    XPX[19][19], XPy[19], beta[19], Xbeta[10000],
    resi[10000], omega=0, sigma0, sigmabeta[19],
    P[10000], p, sumP, pixnr[20000];
  double 	Xp, Yp, Zp, xp, yp, xpd, ypd, r, qq;
  FILE 	*fp1;
  int dummy, multi,numbers;
  double al,be,ga,nGl,e1_x,e1_y,e1_z,e2_x,e2_y,e2_z,n1,n2,safety_x,safety_y,safety_z;



  /* read, which parameters shall be used */
  fp1 = fopen ("parameters/orient.par","r");
  if (fp1){     
      fscanf (fp1,"%d", &useflag);
      fscanf (fp1,"%d", &ccflag);
      fscanf (fp1,"%d", &xhflag);
      fscanf (fp1,"%d", &yhflag);
      fscanf (fp1,"%d", &k1flag);
      fscanf (fp1,"%d", &k2flag);
      fscanf (fp1,"%d", &k3flag);
      fscanf (fp1,"%d", &p1flag);
      fscanf (fp1,"%d", &p2flag);
      fscanf (fp1,"%d", &scxflag);
      fscanf (fp1,"%d", &sheflag);
      fscanf (fp1,"%d", &interfflag);
      fclose (fp1);
  }


  //if(interfflag){
      nGl = sqrt(pow(G0.vec_x,2.)+pow(G0.vec_y,2.)+pow(G0.vec_z,2.));
	  e1_x=2*G0.vec_z-3*G0.vec_x;
	  e1_y=3*G0.vec_x-1*G0.vec_z;
	  e1_z=1*G0.vec_y-2*G0.vec_y;
	  n1=sqrt(pow(e1_x,2.)+pow(e1_y,2.)+pow(e1_z,2.));
	  e1_x=e1_x/n1;
	  e1_y=e1_y/n1;
	  e1_z=e1_z/n1;
	  e2_x=e1_y*G0.vec_z-e1_z*G0.vec_x;
	  e2_y=e1_z*G0.vec_x-e1_x*G0.vec_z;
	  e2_z=e1_x*G0.vec_y-e1_y*G0.vec_y;
	  n2=sqrt(pow(e2_x,2.)+pow(e2_y,2.)+pow(e2_z,2.));
	  e2_x=e2_x/n2;
	  e2_y=e2_y/n2;
	  e2_z=e2_z/n2;
	  al=0;
	  be=0;
	  ga=0;
  //}
  
  
  printf("\n Inside orient_v3, initialize memory \n"); 

  /* init X, y (set to zero) */
  for (i=0; i<5000; i++)
    {
      for (j=0; j<19; j++) {  
        X[i][j] = 0.0;
      }
      y[i] = 0.0;  P[i] = 0.0;
    }

   printf("\n Memory is initialized, \n");
   
  /* init identities */
  ident[0] = I0.cc;  ident[1] = I0.xh;  ident[2] = I0.yh;
  ident[3]=ap0.k1; ident[4]=ap0.k2; ident[5]=ap0.k3;
  ident[6]=ap0.p1; ident[7]=ap0.p2;
  ident[8]=ap0.scx; ident[9]=ap0.she;

  /* main loop, program runs through it, until none of the beta values
     comes over a threshold and no more points are thrown out
     because of their residuals */


  safety_x=G0.vec_x;
  safety_y=G0.vec_y;
  safety_z=G0.vec_z;

  printf("\n\n start iterations, orient_v3 \n");
  itnum = 0;  stopflag = 0;
  while ((stopflag == 0) && (itnum < 80))
    {
      
      itnum++;
      
      printf ("\n\n %2d. iteration \n", itnum);
      
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
	  Xp = fix[i].x;  Yp = fix[i].y;  Zp = fix[i].z;
	  rotation_matrix (Ex0, Ex0.dm);
	  
	  // Debugger print
	  // printf("\n %d %f %f %f %d %d \n",i,Xp,Yp,Zp,crd[i].pnr, fix[i].pnr);
	  
	  img_coord (Xp, Yp, Zp, Ex0, I0, G0, ap0, mm, &xp, &yp);
	  
	  if ((i % 100) == 0){
	   printf("\n %d %f %f %f %d %d %f %f \n",i,Xp,Yp,Zp,crd[i].pnr, fix[i].pnr,xp,yp);
	  }

	  /* derivatives of add. parameters */

	  r = sqrt (xp*xp + yp*yp);

	  X[n][7] = ap0.scx;
	  X[n+1][7] = sin(ap0.she);

	  X[n][8] = 0;
	  X[n+1][8] = 1;

	  X[n][9] = ap0.scx * xp * r*r;
	  X[n+1][9] = yp * r*r;

	  X[n][10] = ap0.scx * xp * pow(r,4.0);
	  X[n+1][10] = yp * pow(r,4.0);

	  X[n][11] = ap0.scx * xp * pow(r,6.0);
	  X[n+1][11] = yp * pow(r,6.0);

	  X[n][12] = ap0.scx * (2*xp*xp + r*r);
	  X[n+1][12] = 2 * xp * yp;

	  X[n][13] = 2 * ap0.scx * xp * yp;
	  X[n+1][13] = 2*yp*yp + r*r;

	  qq =  ap0.k1*r*r; qq += ap0.k2*pow(r,4.0);
	  qq += ap0.k3*pow(r,6.0);
	  qq += 1;
	  X[n][14] = xp * qq + ap0.p1 * (r*r + 2*xp*xp) + 2*ap0.p2*xp*yp;
	  X[n+1][14] = 0;

	  X[n][15] = -cos(ap0.she) * yp;
	  X[n+1][15] = -sin(ap0.she) * yp;



	  /* numeric derivatives */

	  Ex0.x0 += dm;
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][0]      = (xpd - xp) / dm;
	  X[n+1][0] = (ypd - yp) / dm;
	  Ex0.x0 -= dm;

	  Ex0.y0 += dm;
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][1]      = (xpd - xp) / dm;
	  X[n+1][1] = (ypd - yp) / dm;
	  Ex0.y0 -= dm;

	  Ex0.z0 += dm;
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][2]      = (xpd - xp) / dm;
	  X[n+1][2] = (ypd - yp) / dm;
	  Ex0.z0 -= dm;

	  Ex0.omega += drad;
	  rotation_matrix (Ex0, Ex0.dm);
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][3]      = (xpd - xp) / drad;
	  X[n+1][3] = (ypd - yp) / drad;
	  Ex0.omega -= drad;

	  Ex0.phi += drad;
	  rotation_matrix (Ex0, Ex0.dm);
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][4]      = (xpd - xp) / drad;
	  X[n+1][4] = (ypd - yp) / drad;
	  Ex0.phi -= drad;

	  Ex0.kappa += drad;
	  rotation_matrix (Ex0, Ex0.dm);
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][5]      = (xpd - xp) / drad;
	  X[n+1][5] = (ypd - yp) / drad;
	  Ex0.kappa -= drad;

	  I0.cc += dm;
	  rotation_matrix (Ex0, Ex0.dm);
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][6]      = (xpd - xp) / dm;
	  X[n+1][6] = (ypd - yp) / dm;
	  I0.cc -= dm;
      
	  //G0.vec_x += dm;
	  //safety_x=G0.vec_x;
	  //safety_y=G0.vec_y;
	  //safety_z=G0.vec_z;
	  al +=dm;
	  G0.vec_x+=e1_x*nGl*al;G0.vec_y+=e1_y*nGl*al;G0.vec_z+=e1_z*nGl*al;
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][16]      = (xpd - xp) / dm;
	  X[n+1][16] = (ypd - yp) / dm;
	  //G0.vec_x -= dm;
	  //G0.vec_x-=e1_x*nGl*al;G0.vec_y-=e1_y*nGl*al;G0.vec_z-=e1_z*nGl*al;
	  al-=dm;
	  G0.vec_x=safety_x;
	  G0.vec_y=safety_y;
	  G0.vec_z=safety_z;

	  //G0.vec_y += dm;
	  be +=dm;
	  G0.vec_x+=e2_x*nGl*be;G0.vec_y+=e2_y*nGl*be;G0.vec_z+=e2_z*nGl*be;
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][17]      = (xpd - xp) / dm;
	  X[n+1][17] = (ypd - yp) / dm;
	  //G0.vec_y -= dm;
	  //G0.vec_x-=e2_x*nGl*be;G0.vec_y-=e2_y*nGl*be;G0.vec_z-=e2_z*nGl*be;
	  be-=dm;
	  G0.vec_x=safety_x;
	  G0.vec_y=safety_y;
	  G0.vec_z=safety_z;

	  //G0.vec_y += dm;
	  ga +=dm;
	  G0.vec_x+=G0.vec_x*nGl*ga;G0.vec_y+=G0.vec_y*nGl*ga;G0.vec_z+=G0.vec_z*nGl*ga;
	  img_coord (Xp,Yp,Zp, Ex0,I0, G0, ap0, mm, &xpd,&ypd);
	  X[n][18]      = (xpd - xp) / dm;
	  X[n+1][18] = (ypd - yp) / dm;
	  //G0.vec_y -= dm;
	  //G0.vec_x-=G0.vec_x*nGl*ga;G0.vec_y-=G0.vec_y*nGl*ga;G0.vec_z-=G0.vec_z*nGl*ga;
	  ga-=dm;
	  G0.vec_x=safety_x;
	  G0.vec_y=safety_y;
	  G0.vec_z=safety_z;
	  
	  y[n]   = crd[i].x - xp;
	  y[n+1] = crd[i].y - yp;

	  n += 2;
	} // end if crd == fix
      
      
      n_obs = n;
      
      printf(" n_obs = %d\n", n_obs); 


      /* identities */

      for (i=0; i<10; i++)  X[n_obs+i][6+i] = 1;

      y[n_obs+0] = ident[0] - I0.cc;		y[n_obs+1] = ident[1] - I0.xh;
      y[n_obs+2] = ident[2] - I0.yh;		y[n_obs+3] = ident[3] - ap0.k1;
      y[n_obs+4] = ident[4] - ap0.k2;		y[n_obs+5] = ident[5] - ap0.k3;
      y[n_obs+6] = ident[6] - ap0.p1;		y[n_obs+7] = ident[7] - ap0.p2;
      y[n_obs+8] = ident[8] - ap0.scx;		y[n_obs+9] = ident[9] - ap0.she;



      /* weights */
      for (i=0; i<n_obs; i++)  P[i] = 1;
      if ( ! ccflag)  P[n_obs+0] = 1e20;
      if ( ! xhflag)  P[n_obs+1] = 1e20;
      if ( ! yhflag)  P[n_obs+2] = 1e20;
      if ( ! k1flag)  P[n_obs+3] = 1e20;
      if ( ! k2flag)  P[n_obs+4] = 1e20;
      if ( ! k3flag)  P[n_obs+5] = 1e20;
      if ( ! p1flag)  P[n_obs+6] = 1e20;
      if ( ! p2flag)  P[n_obs+7] = 1e20;
      if ( ! scxflag) P[n_obs+8] = 1e20;
      if ( ! sheflag) P[n_obs+9] = 1e20;


      n_obs += 10;  sumP = 0;
      for (i=0; i<n_obs; i++)	       	/* homogenize */
	{
	  p = sqrt (P[i]);
	  for (j=0; j<19; j++)  Xh[i][j] = p * X[i][j];
	  yh[i] = p * y[i];  sumP += P[i];
	}



      /* Gauss Markoff Model */
	  numbers=16;
	  if(interfflag){
         numbers=18;
	  }
	  
	  ata_v2 ((double *) Xh, (double *) XPX, n_obs, numbers, 19 );
      matinv_v2 ((double *) XPX, numbers, 19);
      atl_v2 ((double *) XPy, (double *) Xh, yh, n_obs, numbers, 19);
      matmul_v2 ((double *) beta, (double *) XPX, (double *) XPy, numbers,numbers,1,19,19);
	  
      stopflag = 1;
	  convergeflag = 1;
      //puts ("\n==> beta :\n");
      for (i=0; i<numbers; i++)
	{
	  printf (" beta[%d] = %10.6f\n  ",i, beta[i]);
	  if (fabs (beta[i]) > 0.0001)  stopflag = 0;	/* more iterations */////Achtung
	  if (fabs (beta[i]) > 0.01)  convergeflag = 0;
	}
      //printf ("\n\n");
	  
	  if ( ! ccflag) beta[6]=0;
      if ( ! xhflag) beta[7]=0;
      if ( ! yhflag) beta[8]=0;
      if ( ! k1flag) beta[9]=0;
      if ( ! k2flag) beta[10]=0;
      if ( ! k3flag) beta[11]=0;
      if ( ! p1flag) beta[12]=0;
      if ( ! p2flag) beta[13]=0;
      if ( ! scxflag)beta[14]=0;
      if ( ! sheflag) beta[15]=0;
      
      Ex0.x0 += beta[0];  Ex0.y0 += beta[1];  Ex0.z0 += beta[2];
      Ex0.omega += beta[3];  Ex0.phi += beta[4];  Ex0.kappa += beta[5];
      I0.cc += beta[6];  I0.xh += beta[7];  I0.yh += beta[8];
      ap0.k1 += beta[9];  ap0.k2 += beta[10];  ap0.k3 += beta[11];
      ap0.p1 += beta[12];  ap0.p2 += beta[13];
      ap0.scx += beta[14];  ap0.she += beta[15];
	  if(interfflag){
	  //G0.vec_x += beta[16];	  
	  //G0.vec_y += beta[17];
      G0.vec_x+=e1_x*nGl*beta[16];G0.vec_y+=e1_y*nGl*beta[16];G0.vec_z+=e1_z*nGl*beta[16];
	  G0.vec_x+=e2_x*nGl*beta[17];G0.vec_y+=e2_y*nGl*beta[17];G0.vec_z+=e2_z*nGl*beta[17];
	  //G0.vec_x+=G0.vec_x*nGl*beta[18];G0.vec_y+=G0.vec_y*nGl*beta[18];G0.vec_z+=G0.vec_z*nGl*beta[18];
	  }
	  beta[0]=beta[0];
    } // end of while iterations and stopflag



  /* compute residuals etc. */

  matmul_v2 ( (double *) Xbeta, (double *) X, (double *) beta, n_obs, numbers, 1, n_obs, 19);
  omega = 0;
  for (i=0; i<n_obs; i++)
    {
      resi[i] = Xbeta[i] - y[i];  omega += resi[i] * P[i] * resi[i];
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
  printf ("X0 =    %8.3f   +/- %8.3f\n", Ex0.x0, sigmabeta[0]);
  printf ("Y0 =    %8.3f   +/- %8.3f\n", Ex0.y0, sigmabeta[1]);
  printf ("Z0 =    %8.3f   +/- %8.3f\n", Ex0.z0, sigmabeta[2]);
  printf ("omega = %8.4f   +/- %8.4f\n", Ex0.omega*ro, sigmabeta[3]*ro);
  printf ("phi   = %8.4f   +/- %8.4f\n", Ex0.phi*ro, sigmabeta[4]*ro);
  printf ("kappa = %8.4f   +/- %8.4f\n", Ex0.kappa*ro, sigmabeta[5]*ro);
  if(interfflag){
  printf ("G0.vec_x = %8.4f   +/- %8.4f\n", G0.vec_x/nGl, (sigmabeta[16]+sigmabeta[17]));
  printf ("G0.vec_y = %8.4f   +/- %8.4f\n", G0.vec_y/nGl, (sigmabeta[16]+sigmabeta[17]));
  printf ("G0.vec_z = %8.4f   +/- %8.4f\n", G0.vec_z/nGl, (sigmabeta[16]+sigmabeta[17]));
  }
  //printf ("vec_z = %8.4f   +/- %8.4f\n", G0.vec_z, sigmabeta[18]);
  printf ("camera const  = %8.5f   +/- %8.5f\n", I0.cc, sigmabeta[6]);
  printf ("xh            = %8.5f   +/- %8.5f\n", I0.xh, sigmabeta[7]);
  printf ("yh            = %8.5f   +/- %8.5f\n", I0.yh, sigmabeta[8]);
  printf ("k1            = %8.5f   +/- %8.5f\n", ap0.k1, sigmabeta[9]);
  printf ("k2            = %8.5f   +/- %8.5f\n", ap0.k2, sigmabeta[10]);
  printf ("k3            = %8.5f   +/- %8.5f\n", ap0.k3, sigmabeta[11]);
  printf ("p1            = %8.5f   +/- %8.5f\n", ap0.p1, sigmabeta[12]);
  printf ("p2            = %8.5f   +/- %8.5f\n", ap0.p2, sigmabeta[13]);
  printf ("scale for x'  = %8.5f   +/- %8.5f\n", ap0.scx, sigmabeta[14]);
  printf ("shearing      = %8.5f   +/- %8.5f\n", ap0.she*ro, sigmabeta[15]*ro);


  fp1 = fopen_r ("parameters/examine.par");
  fscanf (fp1,"%d\n", &dummy);
  fscanf (fp1,"%d\n", &multi);
  fclose (fp1);
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
      intx2 = intx1 + resi[i]*5000;
      inty2 = inty1 + resi[i+1]*5000;
 	orient_x1[nr][n]=intx1;
	orient_y1[nr][n]=inty1;
	orient_x2[nr][n]=intx2;
	orient_y2[nr][n]=inty2;
	orient_n[nr]=n;
    }



  if (convergeflag){
      rotation_matrix (Ex0, Ex0.dm);
      *Ex = Ex0;	*I = I0;	*ap = ap0; *G = G0;
  }
  else{	
	  //rotation_matrix (Ex0, Ex0.dm);//////carefullll!!!!
      //*Ex = Ex0;	*I = I0;	*ap = ap0; *G = G0;//////carefullll!!!!
	  puts ("orientation does not converge");
  }
}

