#include <math.h>
#include <stdio.h>
#include <string.h>

#include "epi.h"


int dumbbell_pyptv = 0;

/* temprorary solution, add imgcoord.h */
void img_xy_mm_geo (double X, double Y, double Z, Calibration *cal2, \
mm_np mmp, double *xa, double *ya);

void img_xy_mm_geo (double X, double Y, double Z, Calibration *cal2, \
mm_np mmp, double *xa, double *ya){

}



/*  ray tracing gives the point of exit and the direction
      cosines at the waterside of the glass;
      min. and max. depth give window in object space,
      which can be transformed into _2 image
      (use img_xy_mm because of comparison with img_geo)  
*/

void epi_mm (double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np mmp, volume_par *vpar,
    double *xmin, double *ymin, double *xmax, double *ymax){

  double a, b, c, xa, ya, xb, yb;
  double X1, Y1, Z1, X, Y, Z;
  double Zmin, Zmax;
  double pos[3], v[3]; 
  

  // ray_tracing (x1,y1, Ex1, I1, G1, mmp, &X1, &Y1, &Z1, &a, &b, &c);
  ray_tracing (xl, yl, cal1, mmp, pos, v);
  
  /* convert back into X1,Y1,Z1, a,b,c for clarity */
  X1 = pos[0]; Y1 = pos[1]; Z1 = pos[2];
  a = v[0]; b = v[1]; c = v[2]; 

  /* calculate min and max depth for position (valid only for one setup) */
  Zmin = vpar->Zmin_lay[0]
    + (X1 - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
  Zmax = vpar->Zmax_lay[0]
    + (X1 - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);

  Z = Zmin;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c;
  //img_xy_mm_geo_old (X,Y,Z, Ex2, I2,     mmp, &xa, &ya);
  img_xy_mm_geo (X, Y, Z, cal2, mmp, &xa, &ya);

  Z = Zmax;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c;
  //img_xy_mm_geo_old (X,Y,Z, Ex2, I2,     mmp, &xb, &yb);
  img_xy_mm_geo (X,Y,Z, cal2, mmp, &xb, &yb);

  /*  ==> window given by xa,ya,xb,yb  */

  *xmin = xa;  *ymin = ya;  *xmax = xb;  *ymax = yb;

  // return (0);
}


  /*  ray tracing gives the point of exit and the direction
      cosines at the waterside of the glass;
      min. and max. depth give window in object space,
      which can be transformed into _2 image
      (use img_xy_mm because of comparison with img_geo)  */
      
void epi_mm_2D (double x1, double y1, Calibration *cal1, mm_np mmp, volume_par *vpar, \
double *xp, double *yp, double *zp){

  double a, b, c;
  double X1,Y1,Z1,X,Y,Z;
  double pos[3], v[3];
  double Zmin, Zmax;
  
  /* ray_tracing_v2 (x1,y1, Ex1, I1, G1, mmp, &X1, &Y1, &Z1, &a, &b, &c); */
  ray_tracing (x1, y1, cal1, mmp, pos, v);
  
  
  /* convert back into X1,Y1,Z1, a,b,c for clarity */
  X1 = pos[0]; Y1 = pos[1]; Z1 = pos[2];
  a = v[0]; b = v[1]; c = v[2]; 
  

  /* calculate min and max depth for position (valid only for one setup) */
  Zmin = vpar->Zmin_lay[0]
    + (X1 - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
  Zmax = vpar->Zmax_lay[0]
    + (X1 - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) /
    (vpar->X_lay[1] - vpar->X_lay[0]);

  Z = 0.5*(Zmin+Zmax);   
  X = X1 + (Z-Z1) * a/c;   
  Y = Y1 + (Z-Z1) * b/c;
  
  *xp = X; *yp = Y; *zp = Z;

}

/* find_candidate is searching in the image space of the image all the candidates
around the epipolar line originating from another camera.
Inputs:
	coord_2d structure of a pointer, x,y
	target structure of all the particles in the image: x,y, n, nx, ny, sumg ...
	num - number of particles in the image (int)
	xa,ya,xb,yb - end points of the epipolar line (double, in millimeters)
	n, nx, ny, sumg  - properties of the particle as defined in the parameters, number of 
	pixels in total, size in pixels horizontally and vertically and sum of grey values
	nr - number of the camera
	volume parameters
	control parameters
	Calibration parameters
	
Output: 
	count of the candidates (int pointer)
    candidates structure - pointer, tolerance and correlation value (a sort of a quality)
	
*/

void find_candidate (coord_2d *crd, target *pix, int num, double xa, double ya, \
double xb, double yb, int n, int nx, int ny, int sumg, candidate cand[], int *count, \
int nr, volume_par *vpar, control_par *cpar, Calibration *cal){
/*  binarized search in a x-sorted coord-set, exploits shape information  */
/*  gives messages (in examination)  */

  register int	j;
  int	       	j0, dj, p2;
  double      	m, b, d, temp, qn, qnx, qny, qsumg, corr;
  double       	xmin, xmax, ymin, ymax,particle_size;
  int           dumbbell = 0;
  double 		tol_band_width;
  
  int number_candidates = 4; /* or 8 as it was originally used - might be a memory issue */
  
  /* Alex: decided to make tol_band_width back to eps - it's clear in mm
  then all the following part is irrelevant at the moment, I will remove it out */
  
  tol_band_width = vpar->eps0;
  
  /* 
     Beat Mai 2010 for dumbbell



  if (dumbbell_pyptv==1) dumbbell=1;

  if (dumbbell==0){
	
	  /* Beat version of April 2010 
	  if (nx > ny) particle_size = nx;
	  else       particle_size = ny;
	  */
	  particle_size = MAX(nx,ny);
	  
	  /* tol_band_width means that vpar->eps0 is not in millimeters anymore but in 
	  some relative units: pix_x is millimeters per pixel, times pixel size of the particle
	  gives particle size in millimeters. then we multiply it by eps0 and we get eps0 times
	  the size of the particles
	  */ 
	  
	  /* Alex: decided to make tol_band_width back to eps - it's clear in mm */
	  /* tol_band_width = vpar->eps0 * 0.5 * (cpar->pix_x + cpar->pix_y) * particle_size; */
	  
	  tol_band_width = vpar->eps0;
	  
	  printf("tol_band_width %f \n", tol_band_width);
  }
  else {
  /* for dumbbell it's still the millimiters given in the parameter file */
      tol_band_width = vpar->eps0;
  }
  
  /* this line is therefore also not well defined - whether the limit is on millimeters
  or in parts of the particle size? */
  if (tol_band_width < 0.05 ) tol_band_width = 0.05; /* 50 micron ? */


  /* define sensor format for search interrupt */
  xmin = (-1) * cpar->pix_x * cpar->imx/2;	xmax = cpar->pix_x * cpar->imx/2;
  ymin = (-1) * cpar->pix_y * cpar->imy/2;	ymax = cpar->pix_y * cpar->imy/2;
  xmin -= cal[nr].int_par.xh;	ymin -= cal[nr].int_par.yh;
  xmax -= cal[nr].int_par.xh;	ymax -= cal[nr].int_par.yh;
      
  
  correct_brown_affin (xmin, ymin, cal[nr].added_par, &xmin, &ymin);
  correct_brown_affin (xmax, ymax, cal[nr].added_par, &xmax, &ymax);
  
  printf(" xmin, xmax, ymin, ymax %f %f %f %f \n", xmin, xmax, ymin, ymax);
  
  
  /* instead of number_candidates i found 4 or 8 which was originally used - not sure why.
  it looks like initialization of the structure that grows later up to MAXCAND
  TODO: 
  wouldn't it be wiser to initialize up to MAXCAND? 
  something like:
  struct candidate cand[MAXCAND]; 
  or
  struct candidate *cand = (struct candidate *) malloc (sizeof (struct candidate) * MAXCAND); */
  for (j=0; j<number_candidates; j++)
    {
      cand[j].pnr = -999;  cand[j].tol = -999;  cand[j].corr = -999;
    }


  /* line equation: y = m*x + b */
  if (xa == xb){	
  		xa += 1e-10;
  		printf("\n Warning: using xa == xb in candidate search \n");
   } 
  	
   /* equation of a line */	
   m = (yb-ya)/(xb-xa);  b = ya - m*xa;
  	
  printf (" m, b %f %f \n", m, b);
  
  
  if (xa > xb)
    {
      temp = xa;  xa = xb;  xb = temp;
    }
  if (ya > yb)
    {
      temp = ya;  ya = yb;  yb = temp;
    }

  printf(" xa, xb, ya, yb %f %f %f %f \n", xa,xb,ya,yb);

  if ( (xb>xmin) && (xa<xmax) && (yb>ymin) && (ya<ymax)){ /* sensor area */
    
      /* binary search for start point of candidate search */
      for (j0=num/2, dj=num/4; dj>1; dj/=2){
	  	if (crd[j0].x < (xa - tol_band_width))  j0 += dj;
	  	else  j0 -= dj;
	  }
      
      j0 -= 12;  if (j0 < 0)  j0 = 0;  	/* due to truncation error we might shift to smaller x */

      for (j=j0, *count=0; j<num; j++){ 	/* candidate search */
	
	    if (crd[j].x > xb+tol_band_width)  return;  /* finish search */

	    if ((crd[j].y > ya-tol_band_width) && (crd[j].y < yb+tol_band_width)){
	    
			if ((crd[j].x > xa-tol_band_width) && (crd[j].x < xb+tol_band_width)){
		
			  d = fabs ((crd[j].y - m*crd[j].x - b) / sqrt(m*m+1));
				  if ( d < tol_band_width ){
					  
					  p2 = crd[j].pnr;
					  
					  /* quality of each parameter is a ratio of the values of the 
					  size n, nx, ny and sum of grey values sumg
					  */
					  if (n  < pix[p2].n)      	qn  = (double) n/pix[p2].n;
					  else		       	qn  = (double) pix[p2].n/n;
					  
					  if (nx < pix[p2].nx)	qnx = (double) nx/pix[p2].nx;
					  else		       	qnx = (double) pix[p2].nx/nx;
					  
					  if (ny < pix[p2].ny)	qny = (double) ny/pix[p2].ny;
					  else		       	qny = (double) pix[p2].ny/ny;
					  
					  if (sumg < pix[p2].sumg)
							qsumg = (double) sumg/pix[p2].sumg;
					  else	qsumg = (double) pix[p2].sumg/sumg;


					  /* empirical correlation coefficient
					 from shape and brightness parameters */
					  corr = (4*qsumg + 2*qn + qnx + qny);
					  /* create a tendency to prefer those matches
					     with brighter targets */
					  corr *= ((double) (sumg + pix[p2].sumg));

					if (qn >= vpar->cn && qnx >= vpar->cnx && qny >= vpar->cny && 
						qsumg > vpar->csumg){
			
						if (*count >= MAXCAND){ 
							printf("More candidates than (maxcand): %d\n",*count); 
							return; 
						}
						cand[*count].pnr = p2; /* or it should be here j as in _plus ??? */
						cand[*count].tol = d;
						cand[*count].corr = corr;
						(*count)++;
						printf ("%d %3.0f/%3.1f \n", p2, corr, d*1000);
					}
				}
			}
	    }
	}
      if (*count == 0)  printf ("- - -");
    }
  else  *count = -1;		       	       /* out of sensor area */

}





