#include <math.h>
#include <stdio.h>
#include <string.h>

#include "epi.h"
#include "multimed.h"


int dumbbell_pyptv = 0;

/* temprorary solution, add imgcoord.h and wait for imgcoord.c */
void img_xy_mm_geo (double X, double Y, double Z, Calibration *cal, \
mm_np mmp, int i_cam, mmlut *mmlut, double *xa, double *ya);


void img_xy_mm_geo (double X, double Y, double Z, Calibration *cal, \
mm_np mm, int i_cam, mmlut *mmlut, double *x, double *y){

  double deno;
  Exterior Ex_t;
  double X_t,Y_t,Z_t,cross_p[3],cross_c[3],Xh,Yh,Zh;

  Exterior Ex = cal->ext_par;
  Interior I = cal->int_par;
  Glass G = cal->glass_par;
  
 

  trans_Cam_Point(Ex, mm, G, X, Y, Z, &Ex_t, &X_t, &Y_t, &Z_t, cross_p,cross_c); 
   
  /* 
  multimed_nlay_v2 (Ex_t,Ex,mm,X_t,Y_t,Z_t,&X_t,&Y_t);
  */
  multimed_nlay (&Ex_t, &mm, X_t, Y_t, Z_t, &X_t, &Y_t, i_cam, mmlut);
   
  back_trans_Point(X_t, Y_t, Z_t, mm, G, cross_p, cross_c,&X,&Y,&Z);

  deno = Ex.dm[0][2] * (X-Ex.x0)
    + Ex.dm[1][2] * (Y-Ex.y0)
    + Ex.dm[2][2] * (Z-Ex.z0);

  *x = - I.cc *  (Ex.dm[0][0] * (X-Ex.x0)
		  + Ex.dm[1][0] * (Y-Ex.y0)
		  + Ex.dm[2][0] * (Z-Ex.z0)) / deno;

  *y = - I.cc *  (Ex.dm[0][1] * (X-Ex.x0)
		  + Ex.dm[1][1] * (Y-Ex.y0)
		  + Ex.dm[2][1] * (Z-Ex.z0)) / deno;
}



/*  ray tracing gives the point of exit and the direction
      cosines at the waterside of the glass;
      min. and max. depth give window in object space,
      which can be transformed into _2 image
      (use img_xy_mm because of comparison with img_geo)  
*/

void epi_mm (double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np mmp, volume_par *vpar,
    int i_cam, mmlut *mmlut, double *xmin, double *ymin, double *xmax, double *ymax){

  double a, b, c, xa, ya, xb, yb;
  double X1, Y1, Z1, X, Y, Z;
  double Zmin, Zmax;
  double pos[3], v[3]; 
  

  // ray_tracing (x1,y1, Ex1, I1, G1, mmp, &X1, &Y1, &Z1, &a, &b, &c);
  ray_tracing (xl, yl, cal1, mmp, pos, v);
  

  /* calculate min and max depth for position (valid only for one setup) */
  Zmin = vpar->Zmin_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
    
  
    
  Zmax = vpar->Zmax_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
    
  Z = Zmin;   X = pos[0] + (Z-pos[2]) * v[0]/v[2];   Y = pos[1] + (Z-pos[2]) * v[1]/v[2];
  
  //img_xy_mm_geo_old (X,Y,Z, Ex2, I2,     mmp, &xa, &ya);
  img_xy_mm_geo (X, Y, Z, cal2, mmp, i_cam, mmlut, &xa, &ya);


  Z = Zmax;   X = pos[0] + (Z-pos[2]) * v[0]/v[2];   Y = pos[1] + (Z-pos[2]) * v[1]/v[2];
  
  //img_xy_mm_geo_old (X,Y,Z, Ex2, I2,     mmp, &xb, &yb);
  img_xy_mm_geo (X, Y, Z, cal2, mmp, i_cam, mmlut, &xb, &yb);

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
	count of the candidates (int *)
    candidates structure - pointer, tolerance and correlation value (a sort of a quality)
	
*/

/* very useful discussion on the mailing list brought us to the understanding that this 
function is used twice with two different strategies:
1. for the correspondences_4 we can abuse the pointer and use 

		cand[*count].pnr = j; 
		
    where 'j' is the row number if the geo [] list sorted by x
    later, in correspondences_4 the same list is used in all the cameras and the same 
    index is used for the pointer and the best candidates are taken from the top of the list
    
2. for any other function, where this re-sorting does not occur and the pointer is the correct
information, then we have to use 

	   cand[*count].pnr = p2;
	
*/

void find_candidate (coord_2d *crd, target *pix, int num, double xa, double ya, \
double xb, double yb, int n, int nx, int ny, int sumg, candidate cand[], int *count, \
int nr, volume_par *vpar, control_par *cpar, Calibration *cal, int is_sorted){
/*  binarized search in a x-sorted coord-set, exploits shape information  */
/*  gives messages (in examination)  */

  register int	j;
  int	       	j0, dj, p2;
  double      	m, b, d, temp, qn, qnx, qny, qsumg, corr;
  double       	xmin, xmax, ymin, ymax,particle_size;
  int           dumbbell = 0;
  double 		tol_band_width;
  
  /* the minimum number of candidates to initialise the array 
  at different versions it was 4 or 8, could be up to MAXCAND 
  */
    
  tol_band_width = vpar->eps0;
  
 
  /* define sensor format for search interrupt */
  xmin = (-1) * cpar->pix_x * cpar->imx/2;	xmax = cpar->pix_x * cpar->imx/2;
  ymin = (-1) * cpar->pix_y * cpar->imy/2;	ymax = cpar->pix_y * cpar->imy/2;
  xmin -= cal[nr].int_par.xh;	ymin -= cal[nr].int_par.yh;
  xmax -= cal[nr].int_par.xh;	ymax -= cal[nr].int_par.yh;
      
  
  correct_brown_affin (xmin, ymin, cal[nr].added_par, &xmin, &ymin);
  correct_brown_affin (xmax, ymax, cal[nr].added_par, &xmax, &ymax);
    
  
  /* we need to reset only few first lines to be sure we do not reuse old pointers */
  for (j=0; j<4; j++){ 
      cand[j].pnr = -999;  cand[j].tol = -999;  cand[j].corr = -999;
   }


  /* line equation: y = m*x + b */
  if (xa == xb){ /* the line is a point or a vertical line in this camera */	
  		xb += 1e-10; /* if we use xa += 1e-10, we always switch later */
   } 
  	
   /* equation of a line */	
   m = (yb-ya)/(xb-xa);  b = ya - m*xa;
  	  
  
  if (xa > xb)
    {
      temp = xa;  xa = xb;  xb = temp;
    }
  if (ya > yb)
    {
      temp = ya;  ya = yb;  yb = temp;
    }

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
						/* 
							when called from correspondences_4 is_sorted = 1
						   	when called from mousefunction is_sorted = 0
						*/ 
						if (is_sorted == 1) cand[*count].pnr = j;
						else cand[*count].pnr = p2;
						
						cand[*count].tol = d;
						cand[*count].corr = corr;
						(*count)++;
					}
				}
			}
	    }
	}
      if (*count == 0)  printf ("- - -");
    }
  else  *count = -1;		       	       /* out of sensor area */
}


