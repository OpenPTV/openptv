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
    target sorted_pix[] is the array of targets or detected dots that have an ID (pnr), 
    pixel position, size of the dot, sum of grey values and another identification (tnr)
    the pnr pointer is the row number of the dot in the calibration block file
*/
target* sortgrid (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], int num, 
                int eps, target pix[])
{
  int	       	i, j;
  target       	*sorted_pix;
  vec2d         calib_point;
  double        xp,yp;
      
  /* sorted pix should be just of the same size as fix[] */
  sorted_pix = (target *) malloc(nfix * sizeof(target));

  for (i=0; i<nfix; i++) sorted_pix[i].pnr = -999;

  
  
  /* and search a detected target nearby */
  for (i=0; i<nfix; i++){
  
        img_coord (fix[i], cal, cpar->mm,  &xp, &yp);
        metric_to_pixel (&(calib_point[0]), &(calib_point[1]), xp, yp, cpar);

  
       /* if the point is not touching the image border */ 
      if ( calib_point[0]> -eps  &&  calib_point[1]> -eps  &&  
           calib_point[0] < cpar->imx + eps  && calib_point[1]< cpar->imy + eps){
                    
          /* find the nearest target point */
          j = nearest_neighbour_pix(pix, num, calib_point[0], calib_point[1], eps);
      
          if (j != -999) {              /* if found */
              sorted_pix[i] = pix[j];          /* assign its row number */
              sorted_pix[i].pnr = i;          /* pointer is a row number of a point */
            }
        }
    }
    
    return(sorted_pix);
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
    FILE* fpp = 0;
    int eps = 0;
    
    fpp = fopen(filename, "r");
    if(!fpp) goto handle_error;
    if(fscanf(fpp, "%d\n",  &eps) == 0) goto handle_error;
    fclose (fpp);
    
    return eps;

handle_error:
    printf("Error reading sortgrid parameter from %s\n", filename);
    if (fpp != NULL) fclose (fpp);
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
vec3d* read_calblock(int *num_points, char* filename) {
    FILE* fpp = NULL;
    int	dummy, k = 0;
    vec3d fix;
    vec3d *ret = (vec3d *) malloc(1); /* reallocated every line */
       
    fpp = fopen (filename, "r");
    if (! fpp) {
        printf("Can't open calibration block file: %s\n", filename);
        goto handle_error;
    }
    
    /* Single-pass read-reallocate.*/
    while (fscanf(fpp, "%d %lf %lf %lf\n", &(dummy),&(fix[0]),
            &(fix[1]),&(fix[2])) == 4) 
    {
        ret = (vec3d *) realloc(ret, (k + 1) * sizeof(vec3d));
        vec_copy(ret[k], fix);
        k++;
    }
    
    if (k == 0) {
        printf("Empty of badly formatted file: %s\n", filename);
        goto handle_error;
    }
    
    fclose (fpp);
    *num_points = k;
    return ret;

handle_error:
    if (fpp != NULL) fclose (fpp);
    *num_points = 0;
    free(ret);
    return NULL;
}
