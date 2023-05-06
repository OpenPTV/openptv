#include <math.h>
#include <stdio.h>
#include <string.h>

#include "epi.h"

/*  epi_mm() takes a point in images space of one camera, positions of this 
    and another camera and returns the epipolar line (in millimeter units) 
    that corresponds to the point of interest in the another camera space.
    
    Arguments:
    double xl, yl - position of the point on the origin camera's image space,
        in [mm].
    Calibration *cal1 - position of the origin camera
    Calibration *cal2 - position of camera on which the line is projected.
    mm_np *mmp - pointer to multimedia model of the experiment.
    volume_par *vpar - limits the search in 3D for the epipolar line

    Output:
    xmin,ymin and xmax,ymax - end points of the epipolar line in the "second"
        camera 
*/

void epi_mm (double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np *mmp, volume_par *vpar,
    double *xmin, double *ymin, double *xmax, double *ymax){

    double Zmin, Zmax;
    vec3d pos, v, X; 


    ray_tracing (xl, yl, cal1, *mmp, pos, v);

    /* calculate min and max depth for position (valid only for one setup) */
    Zmin = vpar->Zmin_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);

    Zmax = vpar->Zmax_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);

    move_along_ray(Zmin, pos, v, X);
    flat_image_coord (X, cal2, mmp, xmin, ymin);

    move_along_ray(Zmax, pos, v, X);
    flat_image_coord (X, cal2, mmp, xmax, ymax);
}


/*  epi_mm_2D() is a very degenerate case of the epipolar geometry use.
    It is valuable only for the case of a single camera with multi-media.
    It takes a point in images space of one (single) camera, positions of this 
    camera and returns the position (in millimeter units) inside the 3D space
    that corresponds to the provided point of interest, limited in the middle of 
    the 3D space, half-way between Zmin and Zmax. In purely 2D experiment, with 
    an infinitely small light sheet thickness or on a flat surface, this will 
    mean the point ray traced through the multi-media into the 3D space.  
    
    Arguments:
    double xl, yl - position of the point in the camera image space [mm].
    Calibration *cal1 - position of the camera
    mm_np *mmp - pointer to multimedia model of the experiment.
    volume_par *vpar - limits the search in 3D for the epipolar line.
    
    Output:
    vec3d out - 3D position of the point in the mid-plane between Zmin and 
        Zmax, which are estimated using volume limits provided in vpar.
*/
      
void epi_mm_2D (double xl, double yl, Calibration *cal1, mm_np *mmp, volume_par *vpar, 
    vec3d out){

  vec3d pos, v;
  double Zmin, Zmax;
    
  ray_tracing (xl, yl, cal1, *mmp, pos, v);
      
  Zmin = vpar->Zmin_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmin_lay[1] - vpar->Zmin_lay[0]) / 
    (vpar->X_lay[1] - vpar->X_lay[0]);
    
  Zmax = vpar->Zmax_lay[0]
    + (pos[0] - vpar->X_lay[0]) * (vpar->Zmax_lay[1] - vpar->Zmax_lay[0]) /
    (vpar->X_lay[1] - vpar->X_lay[0]);
        
  move_along_ray(0.5*(Zmin+Zmax), pos, v, out);
}

/* for candidate search */
#define quality_ratio(a,b) ( ((a) < (b)) ? (double) (a)/(b) : (double) (b)/(a) )

/*  find_candidate() is searching in the image space of the image all the 
    candidates around the epipolar line originating from another camera. It is 
    a binary search in an x-sorted coord-set, exploits shape information of the
    particles.
    
    Arguments:
    coord_2d *crd - points to an array of detected-points position information.
        the points must be in flat-image (brown/affine corrected) coordinates
        and sorted by their x coordinate, i.e. ``crd[i].x <= crd[i + 1].x``.
    target *pix - array of target information (size, grey value, etc.) 
        structures. pix[j] describes the target corresponding to 
        (crd[...].pnr == j).
    int num - number of particles in the image.
    double xa, xb, ya, yb - end points of the epipolar line [mm].
    int n, nx, ny - total, and per dimension pixel size of a typical target,
        used to evaluate the quality of each candidate by comparing to typical.
    int sumg - same, for the grey value.
    
    Outputs:
    candidate cand[] - array of candidate properties. The .pnr property of cand
        points to an index in the x-sorted corrected detections array 
        (``crd``).
    
    Extra configuration Arguments:
    volume_par *vpar - observed volume dimensions.
    control_par *cpar - general scene data s.a. image size.
    Calibration *cal - position and other parameters on the camera seeing 
        the candidates.
    
    Returns:
    int count - the number of selected candidates, length of cand array. 
        Negative if epipolar line out of sensor array.
*/
int find_candidate (coord_2d *crd, target *pix, int num, 
    double xa, double ya, double xb, double yb, int n, int nx, int ny, int sumg, 
    candidate cand[], volume_par *vpar, control_par *cpar, Calibration *cal)
{
  register int	j;
  int	       	j0, dj, p2, count = 0;
  double      	m, b, d, temp, qn, qnx, qny, qsumg, corr;
  double       	xmin, xmax, ymin, ymax;
  double 		tol_band_width;
  
  tol_band_width = vpar->eps0;
 
  /* define sensor format for search interrupt */
  xmin = (-1) * cpar->pix_x * cpar->imx/2;	xmax = cpar->pix_x * cpar->imx/2;
  ymin = (-1) * cpar->pix_y * cpar->imy/2;	ymax = cpar->pix_y * cpar->imy/2;
  xmin -= cal->int_par.xh;	ymin -= cal->int_par.yh;
  xmax -= cal->int_par.xh;	ymax -= cal->int_par.yh;
  
  correct_brown_affin (xmin, ymin, cal->added_par, &xmin, &ymin);
  correct_brown_affin (xmax, ymax, cal->added_par, &xmax, &ymax);
    
  /* line equation: y = m*x + b */
  if (xa == xb) { /* the line is a point or a vertical line in this camera */	
  		xb += 1e-10; /* if we use xa += 1e-10, we always switch later */
  }
  	
  /* equation of a line */	
  m = (yb-ya)/(xb-xa);  b = ya - m*xa;	  
  
  if (xa > xb) {
      temp = xa;
      xa = xb;
      xb = temp;
  }
  if (ya > yb) {
      temp = ya;
      ya = yb;
      yb = temp;
  }

  /* If epipolar line out of sensor area, give up. */

  if ( (xb <= xmin) || (xa >= xmax) || (yb <= ymin) || (ya >= ymax)) {
      return -1;
  }
    
  /* binary search for start point of candidate search */
  for (j0 = num/2, dj = num/4; dj > 1; dj /= 2) {
      if (crd[j0].x < (xa - tol_band_width))
          j0 += dj;
      else  
          j0 -= dj;
  }
      
  /* due to truncation error we might shift to smaller x */
  j0 -= 12;  
  if (j0 < 0)  j0 = 0; 

  for (j = j0; j < num; j++) {  /* candidate search */
      	
      /* Since the list is x-sorted, an out of x-bound candidate is after the
         last possible candidate, so stop. */
      if (crd[j].x > xb + tol_band_width) 
          return count; 

      /* Candidate should at the very least be in the epipolar search window
         to be considred. */
      if ((crd[j].y <= ya - tol_band_width) || (crd[j].y >= yb + tol_band_width))
          continue;
      if ((crd[j].x <= xa - tol_band_width) || (crd[j].x >= xb + tol_band_width))
          continue;
		
      /* Only take candidates within a predefined distance from epipolar line. */			
      d = fabs ((crd[j].y - m*crd[j].x - b) / sqrt(m*m+1));
      if (d >= tol_band_width)
          continue;
        
      p2 = crd[j].pnr;
      
      if (p2 >= num) {
          printf("pnr out of range: %d\n", p2);
          return -1;
      }
					  
      /* quality of each parameter is a ratio of the values of the 
         size n, nx, ny and sum of grey values sumg */
      qn = quality_ratio(n, pix[p2].n);
      qnx = quality_ratio(nx, pix[p2].nx);
      qny = quality_ratio(ny, pix[p2].ny);
      qsumg = quality_ratio(sumg, pix[p2].sumg);
            
      /* Enforce minimum quality values  and maximum candidates */
      if (qn < vpar->cn || qnx < vpar->cnx || qny < vpar->cny ||
          qsumg <= vpar->csumg) continue;
      if (count >= MAXCAND){ 
          printf("More candidates than (maxcand): %d\n", count); 
          return count; 
      }
            
      /* empirical correlation coefficient from shape and brightness 
         parameters */
      corr = (4*qsumg + 2*qn + qnx + qny);
            
      /* prefer matches with brighter targets */
      corr *= ((double) (sumg + pix[p2].sumg));
      
      cand[count].pnr = j;
      cand[count].tol = d;
      cand[count].corr = corr;
      count++;
  }
  return count;
}

