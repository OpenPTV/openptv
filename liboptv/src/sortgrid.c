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

#define NMAX 1024
#define DEFAULT_SEARCH_RADIUS 10

#include "sortgrid.h"

/* 
    nearest_pixel_location () converts the positions of 3d points provided for calibration
    in a calibration file (fix) to the image space for each camera in order to present
    those on a screen for user interaction. Used only in "show initial guess" without
    sorting or finding the corresponding points like in sortgrid_man
    Arguments: 
    Calibration *cal points to calibration parameters
    Control *cpar points to control parameters
    nfix is the integer number of points in the calibration text files or number of files 
    if the calibration is multiplane. 
    coord_3d fix[] array of doubles of 3D positions of the calibration target points
    num is the number of detected (by image processing) dots on the calibration image
    i_cam is the integer number of the camera
    Output:
    pixel_pos calib_points[] structure of integer pixel positions (.x, .y) corresponding 
    to the 3D points in structure fix. 
*/    
void nearest_pixel_location (Calibration* cal, control_par *cpar, int nfix, coord_3d fix[], 
pixel_pos calib_points[]){
  int	       	i, j;
  int	       	intx, inty;
  double       	xp, yp;
  target       	old[1024];
  
  
  for (i=0; i<nfix; i++)
    {
        img_coord (fix[i].pos, cal, cpar->mm,  &xp, &yp);
        metric_to_pixel (&xp, &yp, xp, yp, cpar);

/*      previous name - saved for reference until we change the python bindings 
        x_calib[i_cam][i] =(int) xp;  
        y_calib[i_cam][i] = (int) yp;
*/
        calib_points[i].x = (int) xp;
        calib_points[i].y = (int) yp;
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
    target pix[])
{
  int	       	i, j;
  int	       	intx, inty;
  double       	xp, yp;
  int           tmp, eps = 10;
  target       	old[NMAX];
  pixel_pos     calib_points[NMAX];
  
  eps = read_sortgrid_par("parameters/sortgrid.par");
  if (eps == NULL) eps = DEFAULT_SEARCH_RADIUS;
  
  /* copy and re-initialize pixel data before sorting */
  for (i=0; i<num; i++)	old[i] = pix[i];
  
  for (i=0; i<nfix; i++) {
      pix[i].pnr = -999;  pix[i].x = -999;  pix[i].y = -999;
      pix[i].n = 0; pix[i].nx = 0; pix[i].ny = 0;
      pix[i].sumg = 0;
    }
  

  

  /* reproject all calibration plate points into pixel space
     and search a detected target nearby */
  
 
  nearest_pixel_location (cal, cpar, nfix, fix, calib_points);
  
  for (i=0; i<nfix; i++){
       /* if the point is not touching the image border */ 
      if ( calib_points[i].x > -eps  &&  calib_points[i].x > -eps  &&  
            calib_points[i].x < cpar->imx + eps  &&  calib_points[i].y < cpar->imy + eps){
          
          /* find the nearest target point */
          j = nearest_neighbour_pix (old, num, xp, yp, eps);
      
          if (j != -999) { /* if found */
              pix[i] = old[j];          /* assign its row number */
              pix[i].pnr = fix[i].pnr;  /* assign the pointer of a corresponding point */
              /*
              z_calib[i_cam][i] = fix[i].pnr; // Alex, 18.05.11
              printf("z_calib[%d][%d]=%d\n",i_cam,i,z_calib[i_cam][i]);
              */
            }
        }
    }
}


/* 
  nearest_neighbour_pix () searches for the particle in the image space that is 
  the nearest neighbour of a point x,y. The search is within epsilon distance from
  the point.
   
  Arguments: 
  target pix - database of all the particles in a given frame of a given camera
  int num - number of particles in the database pix
  double x,y - position of a point (can be a mouse click or another particle center)
  double eps - a small floating value of epsilon defining the search region around x,y
  Returns:
  int pnr  - a pointer to the nearest neighbour (index of the particle in the structure) 
  or -999 if no particle is found 
  
  moved here from tools.c in the Tcl/Tk version
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



/* read_sortgrid_par() reads a single line, single value parameter file sortgrid.par 

   Arguments:
   char *filename - path to the text file containing the parameters.
   
   Returns:
   integer eps - search radius in pixels for the sortgrid. If reading failed for 
   any reason, returns NULL.
*/
int read_sortgrid_par(char *filename) {
    FILE* fpp;
    int eps;
    
    fpp = fopen(filename, "r");
    if(fscanf(fpp, "%d\n",  &eps) == 0) goto handle_error;
    fclose (fpp);
    
    return eps;

handle_error:
    printf("Error reading sortgrid parameter from %s\n", filename);
    fclose (fpp);
    return NULL;
}