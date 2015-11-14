#include <math.h>
#include <stdio.h>
#include <string.h>

#include "epi.h"

int dumbbell_pyptv = 0;


/*    epi_mm() takes a point in images space of one camera, positions of this 
      and another camera and returns the epipolar line (in millimeter units) 
      that corresponds to the point of interest in the another camera space.
      Arguments:
      xl,yl - position of the point in the specific camera
      Calibration *cal1 - position of the specific camera
      Calibration *cal2 - position of another camera
      mm_np mmp structure of the multimedia model of the experiment
      volume parameters vpar - limits the search in 3D for the epipolar line
      Output:
      xmin,ymin and xmax,ymax - end points of the epipolar line in the "second" camera 
*/

void epi_mm (double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np mmp, volume_par *vpar,
    double *xmin, double *ymin, double *xmax, double *ymax){

    double Zmin, Zmax;
    vec3d pos, v, X; 


    ray_tracing (xl, yl, cal1, mmp, pos, v);

    /* calculate min and max depth for position (valid only for one setup) */
    Zmin = vpar->Zmin_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);

    Zmax = vpar->Zmax_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);

    move_along_ray(Zmin, pos, v, X);
    flat_image_coord (X, cal2, &mmp, xmin, ymin);

    move_along_ray(Zmax, pos, v, X);
    flat_image_coord (X, cal2, &mmp, xmax, ymax);

}


/*    epi_mm_2D() is a very degenerate case of the epipolar geometry use
      it is valuable only for the single camera with multi-media case
      it takes a point in images space of one (single) camera, positions of this 
      camera and returns the position (in millimeter units) inside the 3D space
      that corresponds to the provided point of interest, limited in the middle of 
      the 3D space, half-way between Zmin and Zmax. In purely 2D experiment, with 
      an infinitely small light sheet thickness or on a flat surface, this will mean
      the point ray traced through the multi-media into the 3D space.  
      Arguments:
      xl,yl - position of the point in the specific camera
      Calibration *cal1 - position of the specific camera
      mm_np mmp structure of the multimedia model of the experiment
      volume parameters vpar - limits the search in 3D for the epipolar line
      Output:
      out - vector of 3D position of the point in the mid-plane between Zmin and Zmax, 
      which are estimated using volume limits provided in the volume_par 
*/
      
void epi_mm_2D (double xl, double yl, Calibration *cal1, mm_np mmp, volume_par *vpar, 
    vec3d out){

  vec3d pos, v;
  double Zmin, Zmax;
    
  ray_tracing (xl, yl, cal1, mmp, pos, v);
      
  Zmin = vpar->Zmin_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
    
  Zmax = vpar->Zmax_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) /
    (vpar->X_lay[1] - vpar->X_lay[0]);
        
  move_along_ray(0.5*(Zmin+Zmax), pos, v, out);

}

/* find_candidate() is searching in the image space of the image all the candidates
    around the epipolar line originating from another camera. It is a binary search 
    in a x-sorted coord-set, exploits shape information of the particles.
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
	is_sorted flag that describes if the particle data is sorted or not (read below)
	
    Output: 
	count of the candidates (int *)
    candidates structure - pointer, tolerance and correlation value (a sort of a quality)
    
    *Warning*
    the minimum number of candidates to initialise the array at different versions 
    was 4 or 8, hard-coded and it could be up to MAXCAND which is a global parameter at 
    the moment.
	
*/

/* very useful discussion on the mailing list brought us to the understanding that this 
function is used twice with two different strategies:
1. for the correspondences_4 we can abuse the pointer and use 

		cand[*count].pnr = j; 
		
    where 'j' is the row number if the geo [] list sorted by x
    later, in correspondences_4 the same list is used in all the cameras and the same 
    index is used for the pointer and the best candidates are taken from the top of the 
    list
    
2. for any other function, where this re-sorting does not occur and the pointer is 
    the correct information, then we have to use 

	   cand[*count].pnr = p2;
	
*/

void find_candidate (coord_2d *crd, target *pix, int num, double xa, double ya, \
double xb, double yb, int n, int nx, int ny, int sumg, candidate cand[], int *count, \
int nr, volume_par *vpar, control_par *cpar, Calibration *cal, int is_sorted){


  register int	j;
  int	       	j0, dj, p2;
  double      	m, b, d, temp, qn, qnx, qny, qsumg, corr;
  double       	xmin, xmax, ymin, ymax,particle_size;
  int           dumbbell = 0;
  double 		tol_band_width;
  

    
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
      /* due to truncation error we might shift to smaller x */
      j0 -= 12;  if (j0 < 0)  j0 = 0; 

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


