/****************************************************************************

Routine:	       	sortgrid.c

Author/Copyright:      	Hans-Gerd Maas

Address:	       	Institute of Geodesy and Photogrammetry
		       	ETH - Hoenggerberg
		       	CH - 8093 Zurich

Creation Date:	       	22.6.88

Description:	       	reads objects, detected by detection etc.,
		       	sorts them with respect to the 13 wires of the grid,
		       	detects missing points
		       	and writes the result to  a new file
			
		       	does not work in each imaginable case !
****************************************************************************/


#include "sortgrid.h"

/* just_plot() converts the positions of 3d points provided for calibration in a 
   calibration file (fix) to the image space for each camera in order to present
   those on a screen for user interaction. Used only in "show initial guess" without
   sorting or finding the corresponding points like in sortgrid_man
   Arguments: 
    Calibration *cal points to calibration parameters
    Control *cpar points to control parameters
    nfix is the integer number of points in the calibration text files or number of files
        if the calibration is multiplane. 
    coord_3d fix[] array of doubles of 3d positions of the calibration target points
    num is the number of detected (by image processing) dots on the calibration image
    i_cam is the integer number of the camera
   Output:
    target pix[] is the array of targets or detected dots that have an ID, pixel position,
    size (total, in x or y), sum of grey values, (pnr, x,y, n,nx,ny,sumg, tnr) 
*/    
void just_plot (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], int num,
    target pix[], int i_cam){
  int	       	i, j;
  int	       	intx, inty;
  double       	xp, yp, eps=10.0;
  // target       	old[512]; Alex, 17.09.09, working on Wesleyan data
  target       	old[1024]; 
  
  
  
  
  
  
   printf("Inside just_plot\n");
   ncal_points[n_img]=nfix;
  
  /* reproject all calibration plate points into pixel space
     and search a detected target nearby */
  
  for (i=0; i<nfix; i++)
    {
        img_coord (fix[i].pos, cal, cpar->mm,  &xp, &yp);
        metric_to_pixel (&xp, &yp, xp, yp, cpar);

        /* draw projected points for check purpuses */
        x_calib[n_img][i] =(int) xp;
        y_calib[n_img][i] = (int) yp;
    


	  printf ("coord of point %d: %d, %d\n", i,intx,inty);
      
//      drawcross (interp, intx, inty, cr_sz+1, n_img, "yellow");
  //   draw_pnr (interp, intx, inty, fix[i].pnr, n_img, "yellow");
      
      
    }
}


/* sortgrid_man () is sorting of detected points by back-projection 
   Arguments: 
    Calibration *cal points to calibration parameters
    Control *cpar points to control parameters
    nfix is the integer number of points in the calibration text files or number of files
        if the calibration is multiplane. 
    coord_3d fix[] array of doubles of 3d positions of the calibration target points
    num is the number of detected (by image processing) dots on the calibration image
    i_cam is the integer number of the camera
   Output:
    target pix[] is the array of targets or detected dots that have an ID, pixel position,
    size (total, in x or y), sum of grey values, (pnr, x,y, n,nx,ny,sumg, tnr) 
*/

void sortgrid_man (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], int num,
    target pix[], int i_cam){
  int	       	i, j;
  int	       	intx, inty;
  double       	xp, yp, eps=10.0;
//  target       	old[512];
  target       	old[1024];
  int            x_calib[4][1000], y_calib[4][1000], z_calib[4][1000];
  
  
  
  /* copy and re-initialize pixel data before sorting */
  for (i=0; i<num; i++)	old[i] = pix[i];
  for (i=0; i<nfix; i++)
    {
      pix[i].pnr = -999;  pix[i].x = -999;  pix[i].y = -999;
      pix[i].n = 0; pix[i].nx = 0; pix[i].ny = 0;
      pix[i].sumg = 0;
    }
  
  
  FILE *fpp = fopen ("parameters/sortgrid.par", "r");

  if (fpp) {
    fscanf (fpp, "%lf", &eps);
    printf ("Sortgrid search radius: %.1f pixel (from sortgrid.par)\n",eps);
    fclose (fpp);
  }
  else {
    printf ("parameters/sortgrid.par does not exist, ");
    printf ("using default search radius 10 pixel\n");
  }
  
  
  /* reproject all calibration plate points into pixel space
     and search a detected target nearby */
  
  for (i=0; i<nfix; i++)
    {
      img_coord (fix[i].x, fix[i].y, fix[i].z, cal, cpar->mm, i_cam, &xp,&yp);
      metric_to_pixel (&xp, &yp, xp, yp, cpar);
      
      /* draw projected points for check purpuses */
      
      intx = (int) xp;
      inty = (int) yp;
// added for Python binding
      x_calib[n_img][i]=intx;
      y_calib[n_img][i]=inty;

	  printf ("coord of point %d: %d, %d\n", i,intx,inty);

   // removed for Python binding 
  //    drawcross (interp, intx, inty, cr_sz+1, n_img, "cyan");
        
      if (xp > -eps  &&  yp > -eps  &&  xp < imx+eps  &&  yp < imy+eps)
        {
          // printf("going to find neighbours %d, %d, %3.1f, %3.1f, %3.1f\n", old, num, xp, yp, eps); 
          j = nearest_neighbour_pix (old, num, xp, yp, eps);
      
          if (j != -999)
            {
              pix[i] = old[j];  pix[i].pnr = fix[i].pnr;
              z_calib[i_cam][i] = fix[i].pnr; // Alex, 18.05.11
              printf("z_calib[%d][%d]=%d\n",n_img,i,z_calib[i_cam][i]);
            }
        }
    }
}


/* 
  nearest_neighbour_pix () 
  Arguments: 
    
  originally from tools.c in Tcl/Tk version
*/ 
int nearest_neighbour_pix (target pix[], int num, double x, double y, double eps){
  register int	j;
  int	       	pnr = -999;
  double       	d, dmin=1e20, xmin, xmax, ymin, ymax;

  xmin = x - eps;  xmax = x + eps;  ymin = y - eps;  ymax = y + eps;

  for (j=0; j<num; j++)		    			/* candidate search */
    {
        if (pix[j].y>ymin && pix[j].y<ymax && pix[j].x>xmin && pix[j].x<xmax)
        {
          d = sqrt ((x-pix[j].x)*(x-pix[j].x) + (y-pix[j].y)*(y-pix[j].y));
          if (d < dmin)
            {
              dmin = d; pnr = j;
            }
        }
    }
  return (pnr);
}

