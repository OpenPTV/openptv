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
    vec3d fix[] array of doubles of 3D positions of the calibration target points
    num is the number of detected (by image processing) dots on the calibration image
    i_cam is the integer number of the camera
    Output:
    vec2d calib_points[] structure of floating pixel positions (.x, .y) corresponding 
    to the 3D points in structure fix. 
*/    
void nearest_pixel_location (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], 
vec2d calib_points[]){
  int	       	i, j;
  int	       	intx, inty;
  double       	xp, yp;
  
  for (i=0; i<nfix; i++)
    {
        img_coord (fix[i], cal, cpar->mm,  &xp, &yp);
        metric_to_pixel (&(calib_points[i][0]), &(calib_points[i][1]), xp, yp, cpar);
    }
}


/* sortgrid () is sorting detected target points by back-projection. Three dimensional 
    positions of the dots on the calibration target are provided with the known IDs. The 
    points are back-projected onto the image space and the nearest neighbour dots 
    identified by image processing routines are selected and sorted according to the 
    pre-defined IDs. The one to one correspondence provides a data on which the 
    calibration process is based on. The nearest neighbour search is a primitive 
    minimum distance search within a pre-defined radius (default = 10) read from the 
    `sortgrid.par` parameter file (radius is given in pixels).
    Arguments: 
    Calibration *cal pointer to calibration parameters
    Control *cpar pointer to control parameters
    nfix is the integer number of points in the calibration text file 
    vec3d fix[] structure 3d positions and integer identification pointers of 
    the calibration target points in the calibration file
    num is the number of detected (by image processing) dots on the calibration image
    Output:
    target pix[] is the array of targets or detected dots that have an ID (pnr), pixel 
    position, size of the dot, sum of grey values and another identification (tnr)
*/

void sortgrid (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], int num, 
                int eps, target pix[])
{
  int	       	i, j;
  int	       	intx, inty;
  int           tmp;
  target       	*old;
  vec2d         *calib_points;
    
  calib_points = (vec2d *) malloc(nfix * sizeof(vec2d));
  old = (target *) malloc(num * sizeof(target));

  /* copy and re-initialize pixel data before sorting and remove pointer */
  /* THERE IS A BUG WHEN NUM < NFIX */
  for (i=0; i<num; i++)
  {	
    old[i] = pix[i]; 
    pix[i].pnr = -999;
   }
  

  /* reproject all calibration plate points into pixel space */
  nearest_pixel_location (cal, cpar, nfix, fix, calib_points);
  
  /* and search a detected target nearby */
  for (i=0; i<nfix; i++){
       /* if the point is not touching the image border */ 
      if ( calib_points[i][0]> -eps  &&  calib_points[i][1]> -eps  &&  
           calib_points[i][0] < cpar->imx + eps  && calib_points[i][1]< cpar->imy + eps){
                    
          /* find the nearest target point */
          j = nearest_neighbour_pix(old, num, calib_points[i][0], calib_points[i][1], eps);
      
          if (j != -999) {              /* if found */
              pix[i] = old[j];          /* assign its row number */
              pix[i].pnr = i+1;         /* assign the pointer of a corresponding point */
            }
        }
    }
    
    free(calib_points);
    free(old);
}


/* 
  nearest_neighbour_pix () searches for the particle in the image space that is 
  the nearest neighbour of a point x,y. The search is within epsilon distance from
  the point.
   
  Arguments: 
  target pix - database of all the particles in a given frame of a given camera 
    that have an ID (pnr), pixel  position, size of the dot, sum of grey values 
    and another identification (tnr) for later tracking
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
    int eps = 0;
    
    fpp = fopen(filename, "r");
    if(!fpp) goto handle_error;
    if(fscanf(fpp, "%d\n",  &eps) == 0) goto handle_error;
    fclose (fpp);
    
    return eps;

handle_error:
    printf("Error reading sortgrid parameter from %s\n", filename);
    fclose (fpp);
    return 0;
}

/* read_calblock() reads the calibration block file into the structure of 3D positons and 
    pointers

   Arguments:
   coord_3d fix[] structure 3d positions and integer identification pointers of 
   the calibration target points in the calibration file
   char *filename - path to the text file containing the calibration points.
   
   Returns:
   int number of valid calibration points. If reading failed for any reason, returns NULL.
*/
int read_calblock(vec3d fix[], char* filename) {
    FILE* fpp;
    int	dummy, k = 0;
       
    fpp = fopen (filename, "r");
    if (! fpp) {
        printf("Can't open calibration block file: %s\n", filename);
        goto handle_error;
    }
    
    while (fscanf(fpp, "%d %lf %lf %lf\n", &(dummy),&(fix[k][0]),
            &(fix[k][1]),&(fix[k][2])) == 4) k++;
    
    fclose (fpp);


    if (k == 0) {
        printf("Empty of badly formatted file: %s\n", filename);
        goto handle_error;
      }
    
    fclose (fpp);
	return k;

handle_error:
    if (fpp != NULL) fclose (fpp);
    return 0;
}