#include <math.h>
#include <stdio.h>
#include <string.h>

#include "epi.h"

int dumbbell_pyptv;

/*  ray tracing gives the point of exit and the direction
      cosines at the waterside of the glass;
      min. and max. depth give window in object space,
      which can be transformed into _2 image
      (use img_xy_mm because of comparison with img_geo)  
*/

int  epi_mm(double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np mmp, volume_par *vpar,
    double *xmin, double *ymin, double *xmax, double *ymax){

  double a, b, c, xa,ya,xb,yb;
  double X1,Y1,Z1, X, Y, Z;
  double Zmin, Zmax;

  // ray_tracing (x1,y1, Ex1, I1, G1, mmp, &X1, &Y1, &Z1, &a, &b, &c);
  ray_tracing (x1, y1, &test_cal, test_mm, (double *)X, (double *)v);

  /* calculate min and max depth for position (valid only for one setup) */
  Zmin = vpar->Zmin_lay[0]
    + (X1 - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
  Zmax = vpar->Zmax_lay[0]
    + (X1 - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);

  Z = Zmin;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c;
  //img_xy_mm_geo_old (X,Y,Z, Ex2, I2,     mmp, &xa, &ya);
  img_xy_mm_geo     (X,Y,Z, Ex2, I2, G2, mmp, &xa, &ya);

  Z = Zmax;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c;
  //img_xy_mm_geo_old (X,Y,Z, Ex2, I2,     mmp, &xb, &yb);
  img_xy_mm_geo     (X,Y,Z, Ex2, I2, G2, mmp, &xb, &yb);

  /*  ==> window given by xa,ya,xb,yb  */

  *xmin = xa;  *ymin = ya;  *xmax = xb;  *ymax = yb;

  return (0);
}


  /*  ray tracing gives the point of exit and the direction
      cosines at the waterside of the glass;
      min. and max. depth give window in object space,
      which can be transformed into _2 image
      (use img_xy_mm because of comparison with img_geo)  */
      
int epi_mm_2D (double x1, double y1, Calibration *cal1, mm_np mmp, volume_par *vpar, \
double *xp, double *yp, double *zp){



  double a, b, c;
  double X1,Y1,Z1,X,Y,Z;
  
  double Zmin, Zmax;

  ray_tracing_v2 (x1,y1, Ex1, I1, G1, mmp, &X1, &Y1, &Z1, &a, &b, &c);

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
  
  *xp=X; *yp=Y; *zp=Z;

  return (0);
}

void find_candidate_plus (crd, pix, num, xa,ya,xb,yb, n, nx, ny, sumg,
						  cand, count, nr, vpar)
/*  binarized search in a x-sorted coord-set, exploits shape information  */

coord_2d	crd[];
target		pix[];
int    		num, *count;
double		xa, ya, xb, yb;
int    		n, nx, ny, sumg;
candidate	cand[];
int	       	nr;	       	/* image number for ap etc. */
volume_par *vpar;
//const char** argv;


{
  register int	j;
  int dummy;
  int	       	j0, dj, p2;
  double      	m, b, d, temp, qn, qnx, qny, qsumg, corr;
  double       	xmin, xmax, ymin, ymax,particle_size;
  int dumbbell=0;
  double tol_band_width;
  
//Beat Mai 2010 for dumbbell

    if (dumbbell_pyptv==1){    dumbbell=1;
  }

  if (dumbbell==0){
	
	  /////here is new Beat version of April 2010
	  if (nx>ny) particle_size=nx;
	  else       particle_size=ny;
	  tol_band_width = vpar->eps0*0.5*(pix_x + pix_y)*particle_size;
  }
  else{
      tol_band_width = vpar->eps0;
  }
  if(tol_band_width<0.05){
       tol_band_width=0.05;
  }

  /* define sensor format for search interrupt */
  xmin = (-1) * pix_x * imx/2;	xmax = pix_x * imx/2;
  ymin = (-1) * pix_y * imy/2;	ymax = pix_y * imy/2;
  xmin -= I[nr].xh;	ymin -= I[nr].yh;
  xmax -= I[nr].xh;	ymax -= I[nr].yh;
  correct_brown_affin (xmin,ymin, ap[nr], &xmin,&ymin);
  correct_brown_affin (xmax,ymax, ap[nr], &xmax,&ymax);

  for (j=0; j<4; j++)
    {
      cand[j].pnr = -999;  cand[j].tol = -999;  cand[j].corr = -999;
    }


  /* line equation: y = m*x + b */
  if (xa == xb)	xa += 1e-10;
  m = (yb-ya)/(xb-xa);  b = ya - m*xa;

  if (xa > xb)
    {
      temp = xa;  xa = xb;  xb = temp;
    }
  if (ya > yb)
    {
      temp = ya;  ya = yb;  yb = temp;
    }

  if ( (xb>xmin) && (xa<xmax) && (yb>ymin) && (ya<ymax))  /* sensor area */
    {
      /* binarized search for start point of candidate search */
      for (j0=num/2, dj=num/4; dj>1; dj/=2)
	{
	  if (crd[j0].x < (xa - tol_band_width))  j0 += dj;
	  else  j0 -= dj;
	}
      j0 -= 12;  if (j0 < 0)  j0 = 0;		       	/* due to trunc */

      for (j=j0, *count=0; j<num; j++)			/* candidate search */
	{
	  if (crd[j].x > xb+tol_band_width)  return;		/* finish search */

	  if ((crd[j].y > ya-tol_band_width) && (crd[j].y < yb+tol_band_width))
	    {
	      if ((crd[j].x > xa-tol_band_width) && (crd[j].x < xb+tol_band_width))
		{
		  d = fabs ((crd[j].y - m*crd[j].x - b) / sqrt(m*m+1));
          
		  /* Beat: modified in April 2010 to allow for better treatment of 
		  //different sized traced particles, in particular colloids and tracers
		  if ( d < eps ){
          */
          /////here is new Beat version of April 2010
		   //if (nx>ny) particle_size=nx;
		   //else       particle_size=ny;
		   if ( d < tol_band_width ){
		   ///////end of new Beat version

		      p2 = crd[j].pnr;
		      if (n  < pix[p2].n)      	qn  = (double) n/pix[p2].n;
		      else		       	qn  = (double) pix[p2].n/n;
		      if (nx < pix[p2].nx)	qnx = (double) nx/pix[p2].nx;
		      else		       	qnx = (double) pix[p2].nx/nx;
		      if (ny < pix[p2].ny)	qny = (double) ny/pix[p2].ny;
		      else		       	qny = (double) pix[p2].ny/ny;
		      if (sumg < pix[p2].sumg)
			        qsumg = (double) sumg/pix[p2].sumg;
		      else	qsumg = (double) pix[p2].sumg/sumg;

		      // empirical correlation coefficient
			  // from shape and brightness parameters 
		      corr = (4*qsumg + 2*qn + qnx + qny);
		      // create a tendency to prefer those matches
			  // with brighter targets 
		      corr *= ((double) (sumg + pix[p2].sumg));

		      if (qn >= vpar->cn && qnx >= vpar->cnx && \
                 qny >= vpar->cny && qsumg > vpar->csumg) {

				 if ( *count < maxcand) {
			        cand[*count].pnr = j;
			        cand[*count].tol = d;
			        cand[*count].corr = corr;
			        (*count)++;
		         } else {
			        dummy=(int)maxcand;
			        printf("in find_candidate_plus: count > maxcand\n");}
			     }
		      }
		   }
           
           
	    }
	}
    }

  else  *count = -1;	 	/* out of sensor area */
}




void find_candidate_plus_msg (crd, pix, num, xa,ya,xb,yb, n, nx, ny, sumg,
							  cand, count, i12, vpar)

/*  binarized search in a x-sorted coord-set, exploits shape information  */
/*  gives messages (in examination)  */

coord_2d	crd[];
target		pix[];
int    		num, *count, i12;
double		xa, ya, xb, yb;
int    		n, nx, ny, sumg;
volume_par *vpar;
/*
candidate	cand[3];
*/
candidate	cand[];

{
  register int	j;
  int	       	j0, dj, p2;
  double        m, b, d, temp, qn, qnx, qny, qsumg, corr;
  double       	xmin, xmax, ymin, ymax;
  double tol_band_width,particle_size;

  /* define sensor format for search interrupt */
  xmin = (-1) * pix_x * imx/2;	xmax = pix_x * imx/2;
  ymin = (-1) * pix_y * imy/2;	ymax = pix_y * imy/2;
  xmin -= I[i12].xh;	ymin -= I[i12].yh;
  xmax -= I[i12].xh;	ymax -= I[i12].yh;
  correct_brown_affin (xmin,ymin, ap[i12], &xmin,&ymin);
  correct_brown_affin (xmax,ymax, ap[i12], &xmax,&ymax);

  if (nx>ny) particle_size=nx;
  else       particle_size=ny;
  tol_band_width = vpar->eps0*0.5*(pix_x + pix_y)*particle_size;

  for (j=0; j<4; j++)
    {
      cand[j].pnr = -999;  cand[j].tol = 999;
    }
  m = (yb-ya)/(xb-xa);  b = ya - m*xa;   /* line equation: y = m*x + b */

  if (xa > xb)
    {
      temp = xa;  xa = xb;  xb = temp;
    }
  if (ya > yb)
    {
      temp = ya;  ya = yb;  yb = temp;
    }

  if ( (xb>xmin) && (xa<xmax) && (yb>ymin) && (ya<ymax)) /* sensor area */
    {
      /* binarized search for start point of candidate search */
      for (j0=num/2, dj=num/4; dj>1; dj/=2)
	{
	  if (crd[j0].x < (xa - tol_band_width))  j0 += dj;
	  else  j0 -= dj;
	}
      j0 -= 12;  if (j0 < 0)  j0 = 0;  	/* due to trunc */

      for (j=j0, *count=0; j<num; j++) 	/* candidate search */
	{
	  if (crd[j].x > xb+tol_band_width)  return;      	/* finish search */

	  if ((crd[j].y > ya-tol_band_width) && (crd[j].y < yb+tol_band_width))
	    {
	      if ((crd[j].x > xa-tol_band_width) && (crd[j].x < xb+tol_band_width))
		{
		  d = fabs ((crd[j].y - m*crd[j].x - b) / sqrt(m*m+1));
          if ( d < tol_band_width ){
		      p2 = crd[j].pnr;
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
                qsumg > vpar->csumg)
			{
			  if (*count>=maxcand)
			    { printf("More candidates than (maxcand): %d\n",*count); return; }
			  cand[*count].pnr = p2;
			  cand[*count].tol = d;
 			  cand[*count].corr = corr;
			  (*count)++;
			  printf ("%d %3.0f/%3.1f \n", p2, corr, d*1000);
			}
		    }
		}
	    }
	}
      if (*count == 0)  puts ("- - -");
    }
  else  *count = -1;		       	       /* out of sensor area */

}





