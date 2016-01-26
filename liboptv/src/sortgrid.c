/****************************************************************************

Routine:	       	sortgrid.c

Adopted from the original sortgrid.c by Maas for the OpenPTV liboptv by 
Alex Liberzon, the sortgrid is rewritten to allow for one-to-one best pair 
assignment using the minimal Euclidean distance. 

Copyright (c) 2015 OpenPTV team
****************************************************************************/

#include "sortgrid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FLAG 999

/* 
    pixel_location () converts the positions of 3d points provided for calibration
    in a calibration file (fix) to the image space for each camera in order to present
    those on a screen for user interaction. Used only in "show initial guess" wit
    hout
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
void pixel_location (Calibration* cal, control_par *cpar, int nfix, vec3d fix[], 
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
    pre-defined IDs. The one to one correspondence is established using the sorted list of 
    distances between the pairs of calibration points and identified pixel positions and 
    descending order assignment.  
    
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
  int	       	i, j, k, pos, size = num*nfix;
  vec2d         *calib_points;
  double        mindist;
  double        *dist_array;
  int           *index;
  double        *array;
    
  calib_points = (vec2d *) malloc(nfix * sizeof(vec2d));
  dist_array = (double *) malloc(size*sizeof(double));
  index = (int *) malloc(size*sizeof(int));
  
    
   /* remove the old pointers for safety */
   for (i=0; i<num; i++) 
   {
        pix[i].pnr = -FLAG;
    }
    
    /* prepare the index array and the distance array, both 1D representations of 2D */
    for (i=0; i<size; i++) 
   {
        index[i] = i;
        dist_array[i] = FLAG;
    }
        

  /* reproject all calibration plate points into pixel space */
  pixel_location (cal, cpar, nfix, fix, calib_points);
  
  /* create the array of distances between every pair of points and store the indices */
  for (i=0; i<nfix; i++)
  {
       /* if the point is not touching the image border */ 
      if ( calib_points[i][0]> -eps &&  calib_points[i][1]> -eps  &&  
           calib_points[i][0] < cpar->imx + eps  && calib_points[i][1]< cpar->imy + eps)
      {
         for (j=0; j<num; j++)
         {
            pos = i*num + j;
            dist_array[pos] = dist(calib_points[i],pix[j],eps);
            index[pos] = pos;
            
          }
       } 
    }  // end of double loop that estimates all the distances and stores the index             
        
    /* now we use the distance matrix to assign target points to the calibration points:
      1. sort the distance array, preserving the index to the original pairs
      2. iterate by assigning pointers and removing the assigned pair from the list of options
   */
   
     index = sorted_order(dist_array, size);
 
     for (pos=0; pos<size; pos++){ //going from top to bottom
       if (dist_array[index[pos]] != FLAG){
         i = index[pos]/num;
         j = index[pos]%num;
         pix[j].pnr = i;
         for (k=pos; k<size; k++){ // from the present position downwards
            if (index[k]/num == i || index[k]%num == j){ 
                dist_array[index[k]] = FLAG;   // flag out the used dots/points
            }
          }  
       }
    }
         
    free(calib_points);
}

/* dist() measures the double value of Euclidean distance between two points in 2D 
    vec2d calib_point - element of calibration points array [0] = x , [1] = y 
    target pix        - element of the array of identified dots in the image
    double eps        - region of search, if the pair is far then eps, its distance is 
                        flagged out using FLAG
*/
double dist (vec2d calib_point, target pix, double eps)
{
    double temp;
    temp = sqrt((calib_point[0]-pix.x)*(calib_point[0]-pix.x) + 
            (calib_point[1]-pix.y)*(calib_point[1]-pix.y));
    if (temp > eps) temp = FLAG; 
    return (temp);
}

/* swap() - swaps the positions of two values in an array using pointers *a, *b */ 
void swap(int *a, int *b)
{
   int temp;
 
   temp = *b;
   *b   = *a;
   *a   = temp;   
}

/* sorted_order() uses swap() function to perform bubble sort on the double array of 
   length n
   Arguments:
   double *arr - array to be sorted
   int n       - length of the array 
   Output:
   double *arr is sorted in the ascending order 
*/
int *sorted_order (const double *arr, int n)
{
  int *idx, i, j;
 
  idx = malloc (sizeof (int) * n);
 
  for (i=0; i<n; i++) idx[i] = i;
 
  for (i=0; i<n; i++)
  {
    for (j=i+1; j<n; j++)
    {
      if (arr[idx[i]] > arr[idx[j]])
      {
        swap (&idx[i], &idx[j]);
      }
    }
  }
 
  return idx;
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