/****************************************************************************

Routine:	       	correspondences.c

Author/Copyright:      	Hans-Gerd Maas

Address:	      	Institute of correcteddesy and Photogrammetry
	      		ETH - Hoenggerberg
	      		CH - 8093 Zurich

Creation Date:	       	1988/89

Description:	       	establishment of correspondences for 2/3/4 cameras

****************************************************************************/

#include <stdio.h>
#include "correspondences.h"


// #include "tools.h"

/* quicksort for list of correspondences in order of match quality */
/* 4 camera version */


/* quicksort_con is helper function to run the 
   qs_con() when the left = 0, right = num - 1
   Arguments:
   pointer to the n_tupel array of candidates for correspondence
    important to hold the .corr property that is sorted in this function
    uses quicksort algorithm https://en.wikipedia.org/wiki/Quicksort
   integer num length of the array
*/
void quicksort_con (n_tupel	*con, int num){
  qs_con (con, 0, num-1);
}

void qs_con (n_tupel *con, int left, int right){
  register int	i, j;
  double       	xm;
  n_tupel      	temp;

  i = left;	j = right;	xm = con[(left+right)/2].corr;

  do
    {
      while (con[i].corr > xm  &&  i<right)	i++;
      while (xm > con[j].corr  &&  j>left)	j--;

      if (i <= j)
	{
	  temp = con[i];
	  con[i] = con[j];
	  con[j] = temp;
	  i++;	j--;
	}
    }
  while (i <= j);

  if (left < j)	 qs_con (con, left, j);
  if (i < right) qs_con (con, i, right);
}


/* quicksort_target_y() uses quicksort algorithm to 
   sort targets in y-order by calling the function
   qs_target_y(target *pix, 0, num-1); see below
   
   Arguments:
   pointer to an array of targets *pix
   integer num equal to the length of the array pix
*/

void quicksort_target_y (target *pix, int num) {
  qs_target_y (pix, 0, num-1);
}

/* qs_target_y() uses quicksort algorithm to 
   sort targets in y-order.
   Arguments:
   pointer to an array of targets *pix
   left, right are integers positions in the
   array. To sort the complete array, use
   qs_target_y(target *pix, 0, len(pix)-1)
   according to https://en.wikipedia.org/wiki/Quicksort
   Note: 
   y in OpenPTV is the vertical direction, 
   x is from left to right and z towards the camera
*/
void qs_target_y (target *pix, int left, int right){
  register int	i, j;
  double ym;
  target temp;

  i = left;	j = right;	ym = pix[(left+right)/2].y;

  do {
      while (pix[i].y < ym  &&  i<right)	i++;
      while (ym < pix[j].y  &&  j>left)	j--;

      if (i <= j) {
        temp = pix[i];
        pix[i] = pix[j];
        pix[j] = temp;
        i++; j--;
      }
  } while (i <= j);

  if (left < j)	qs_target_y (pix, left, j);
  if (i < right) qs_target_y (pix, i, right);
}

/* quicksort_coord2d_x() uses quicksort algorithm to 
   sort coordinate array in x-order by calling the function
   qs_coord2d_x (crd, left, right) see below
   
   Arguments:
   pointer to an array of coordinates 
   integer num equal to the length of the array pix
*/
void quicksort_coord2d_x (coord_2d *crd, int num) {
	qs_coord2d_x (crd, 0, num-1);
}

void qs_coord2d_x (coord_2d	*crd, int left, int right){
	register int	i, j;
	double			xm;
	coord_2d		temp;

	i = left;	j = right;	xm = crd[(left+right)/2].x;

	do
	{
		while (crd[i].x < xm  &&  i<right)	i++;
		while (xm < crd[j].x  &&  j>left)	j--;

		if (i <= j)
		{
			temp = crd[i];
			crd[i] = crd[j];
			crd[j] = temp;
			i++;	j--;
		}
	}
	while (i <= j);

	if (left < j)	qs_coord2d_x (crd, left, j);
	if (i < right)	qs_coord2d_x (crd, i, right);
}

/****************************************************************************/
/*--------------- 4 camera model: consistent quadruplets -------------------*/
/****************************************************************************/

// int correspondences (target pix[][nmax], coord_2d corrected[][nmax], int num[], 
//     volume_par *vpar, control_par *cpar, Calibration calib[], n_tupel *con, 
//     int match_counts[]){
n_tupel *correspondences (frame *frm, volume_par *vpar, control_par *cpar, 
Calibration **calib, int match_counts[]) {
  int 	i,j,k,l,m,n,o,i1,i2,i3,cam,part;
  int   count, match=0, match0=0, match4=0, match3=0, match2=0, match1=0;
  int 	p1,p2,p3,p4, p31, p41, p42;
  int  	pt1;
  double       	xa12,ya12,xb12,yb12,X,Y,Z;
  double       	corr;
  candidate   	cand[MAXCAND];
  n_tupel     	*con0, *con;
  correspond  	*list[4][4];
  coord_2d **corrected;
  double img_x, img_y; /* image center */  
  int **tim;

   con0 = (n_tupel *) malloc(cpar->num_cams*nmax * sizeof(n_tupel));
   con = (n_tupel *) malloc(cpar->num_cams*nmax * sizeof(n_tupel)); 
     
    
    tim = malloc(cpar->num_cams * sizeof(int *));
    if(tim == NULL)
        {
        fprintf(stderr, "out of memory\n");
        return NULL;
        }
    for(i = 0; i < nmax; i++)
        {
        tim[i] = malloc(nmax * sizeof(int));
        if(tim[i] == NULL)
            {
            fprintf(stderr, "out of memory\n");
            return NULL;
            }
        }
  
  /* allocate memory for lists of correspondences */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++){
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++){
       list[i1][i2] = (correspond *) malloc (frm->num_targets[i1] * sizeof (correspond));
       if (list[i1][i2] == NULL){
            fprintf(stderr, "list is not allocated");
            return NULL;
        }
     }
    }



  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
      for (i=0; i<frm->num_targets[i1]; i++)
	{
	  list[i1][i2][i].p1 = 0;
	  list[i1][i2][i].n = 0;
	}

/* if I understand correctly, the number of matches cannot be more than the number of
   targets (dots) in the first image. In the future we'll replace it by the maximum
   number of targets in any image (if we will implement the cyclic search) but for 
   a while we always start with the cam1
*/     	
  for (i = 0; i < nmax; i++) {
    for (j = 0; j < cpar->num_cams; j++) {
        tim[j][i] = 0;
        con0[i].p[j] = -1;
    }
    con0[i].corr = 0.0;
  }
  
    


/*  We work on distortion-corrected image coordinates of particles.
        This loop does the correction. It also recycles the iteration on
        frm->num_cams to allocate some arrays needed later and do some related
        preparation. 
*/
    
    corrected = (coord_2d **) malloc(cpar->num_cams * sizeof(coord_2d *));    
    for (cam = 0; cam < cpar->num_cams; cam++) {
        corrected[cam] = (coord_2d *) malloc(
            frm->num_targets[cam] * sizeof(coord_2d));
        if (corrected[cam] == NULL){
            fprintf(stderr, "corrected is not allocated");
            return NULL;
        }
            
            
        for (part = 0; part < frm->num_targets[cam]; part++) {
        printf("%f %f \n", frm->targets[cam][part].x, frm->targets[cam][part].y);
            pixel_to_metric(&corrected[cam][part].x, 
                            &corrected[cam][part].y,
                            frm->targets[cam][part].x, 
                            frm->targets[cam][part].y,
                            cpar);
                            
            printf("%f %f \n", corrected[cam][part].x, corrected[cam][part].y);
            
            img_x = corrected[cam][part].x - calib[cam]->int_par.xh;
            img_y = corrected[cam][part].y - calib[cam]->int_par.yh;
            
            correct_brown_affin (img_x, img_y, calib[cam]->added_par,
               &corrected[cam][part].x, &corrected[cam][part].y);
            
            corrected[cam][part].pnr = frm->targets[cam][part].pnr;
            
            // printf('%f %f %d \n', corrected[cam][part].x, corrected[cam][part].y, corrected[cam][part].pnr);
            printf("%f %f %d \n", corrected[cam][part].x, corrected[cam][part].y, corrected[cam][part].pnr);
        }
        
        /* This is expected by find_candidate_plus() */
        quicksort_coord2d_x(corrected[cam], frm->num_targets[cam]);
    }
 
/*   matching  1 -> 2,3,4  +  2 -> 3,4  +  3 -> 4 */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++) {

      for (i=0; i<frm->num_targets[i1]; i++)	if (corrected[i1][i].x != -999) {
          epi_mm (corrected[i1][i].x, corrected[i1][i].y, calib[i1], calib[i2], 
          cpar->mm, vpar, &xa12, &ya12, &xb12, &yb12);
          
          // printf('%f %f\n', corrected[i1][i].x, corrected[i1][i].y);
          // printf('%f %f %f %f\n', xa12, ya12, xb12, yb12);
          
	  
          /* origin point in the list */
	      p1 = i;  list[i1][i2][p1].p1 = p1;	pt1 = corrected[i1][p1].pnr;

           /* search for a conjugate point in corrected[i2] */
            count = find_candidate (corrected[i2], frm->targets[i2], frm->num_targets[i2],
            xa12, ya12, xb12, yb12, 
            frm->targets[i1][pt1].n,frm->targets[i1][pt1].nx,frm->targets[i1][pt1].ny,
            frm->targets[i1][pt1].sumg, cand, vpar, cpar, calib[i2]);
             
/* 	         write all corresponding candidates to the preliminary list 
 	          of correspondences */
	  if (count > MAXCAND)	{ count = MAXCAND; }
	  for (j=0; j<count; j++)
	    {
	      list[i1][i2][p1].p2[j] = cand[j].pnr;
	      list[i1][i2][p1].corr[j] = cand[j].corr;
	      list[i1][i2][p1].dist[j] = cand[j].tol;
	    }
	  list[i1][i2][p1].n = count;
	}
  }
  
    /* Image coordinates not needed beyond this point. */
    for (cam = 0; cam < frm->num_cams; cam++) {
        free(corrected[cam]);
    }
    free(corrected);

/*   search consistent quadruplets in the list */
  if (cpar->num_cams == 4) {
      for (i=0, match0=0; i<frm->num_targets[0]; i++)
	{
	  p1 = list[0][1][i].p1;
	  for (j=0; j<list[0][1][i].n; j++)
	    for (k=0; k<list[0][2][i].n; k++)
	      for (l=0; l<list[0][3][i].n; l++)
		{
		  p2 = list[0][1][i].p2[j];
		  p3 = list[0][2][i].p2[k];
		  p4 = list[0][3][i].p2[l];
		  for (m=0; m<list[1][2][p2].n; m++) {
			p31 = list[1][2][p2].p2[m];
            if (p3 != p31) continue;
		    for (n=0; n<list[1][3][p2].n; n++)
		      {
			p41 = list[1][3][p2].p2[n];
			if (p4 != p41) continue;
              i3 = list[2][3][p3].n;
			  for (o=0; o<i3; o++)
			    {
			      p42 = list[2][3][p3].p2[o];
			      if (p4 == p42)
				{
				  corr = (list[0][1][i].corr[j]
					  + list[0][2][i].corr[k]
					  + list[0][3][i].corr[l]
					  + list[1][2][p2].corr[m]
					  + list[1][3][p2].corr[n]
					  + list[2][3][p3].corr[o])
				    / (list[0][1][i].dist[j]
				       + list[0][2][i].dist[k]
				       + list[0][3][i].dist[l]
				       + list[1][2][p2].dist[m]
				       + list[1][3][p2].dist[n]
				       + list[2][3][p3].dist[o]);
				  if (corr > vpar->corrmin)
				    {
				      /*accept as preliminary match */
				      con0[match0].p[0] = p1;
				      con0[match0].p[1] = p2;
				      con0[match0].p[2] = p3;
				      con0[match0].p[3] = p4;
				      con0[match0++].corr = corr;
				      if (match0 == 4*nmax)	/* security */
					{
					  printf ("Overflow in correspondences:");
					  printf (" > %d matches\n", match0);
					  i = frm->num_targets[0];
					}
				    }
				}
			    }
              
		      }
            }
		}
	}

/*       sort quadruplets for match quality (.corr) */
        quicksort_con (con0, match0);

/*       take quadruplets from the top to the bottom of the sorted list
       only if none of the points has already been used */
      for (i=0, match=0; i<match0; i++)
	{
	  p1 = con0[i].p[0];	if (p1 > -1)	if (++tim[0][p1] > 1)	continue;
	  p2 = con0[i].p[1];	if (p2 > -1)	if (++tim[1][p2] > 1)	continue;
	  p3 = con0[i].p[2];	if (p3 > -1)	if (++tim[2][p3] > 1)	continue;
	  p4 = con0[i].p[3];	if (p4 > -1)	if (++tim[3][p4] > 1)	continue;
	  con[match++] = con0[i];
	}
 
       match4 = match;
     }

/*   search consistent triplets :  123, 124, 134, 234 */
  if ((cpar->num_cams == 4 && cpar->allCam_flag == 0) || cpar->num_cams == 3)
    {
      //printf("Search consistent triplets \n");
      match0 = 0;
      for (i1 = 0; i1 < cpar->num_cams - 2; i1++)
        for (i2 = i1 + 1; i2 < cpar->num_cams - 1; i2++)
           for (i3 = i2 + 1; i3 < cpar->num_cams; i3++)
				for (i=0; i<frm->num_targets[i1]; i++){
					 p1 = list[i1][i2][i].p1;
					 if (p1 > nmax  ||  tim[i1][p1] > 0)	continue;

					 for (j=0; j<list[i1][i2][i].n; j++)
					  for (k=0; k<list[i1][i3][i].n; k++) {
						  p2 = list[i1][i2][i].p2[j];
						  if (p2 > nmax  ||  tim[i2][p2] > 0)	continue;
						  p3 = list[i1][i3][i].p2[k];
						  if (p3 > nmax  ||  tim[i3][p3] > 0)	continue;
						  						  
						  for (m=0; m<list[i2][i3][p2].n; m++) {
						     p31 = list[i2][i3][p2].p2[m];
						     
						     if (p3 == p31) {
							    corr = (list[i1][i2][i].corr[j]
								  + list[i1][i3][i].corr[k]
								  + list[i2][i3][p2].corr[m])
							    / (list[i1][i2][i].dist[j]
							    + list[i1][i3][i].dist[k]
							    + list[i2][i3][p2].dist[m]);
							  if (corr > vpar->corrmin) {
								for (n = 0; n < cpar->num_cams; n++) con0[match0].p[n] = -2;
									con0[match0].p[i1] = p1;
									con0[match0].p[i2] = p2;
									con0[match0].p[i3] = p3;
									con0[match0++].corr = corr;
								}
								if (match0 == 4*nmax) {   /* security */
									printf ("Overflow in correspondences:\n");
									printf (" > %d matches\n", match0);
									i = frm->num_targets[i1]; /* Break out of the outer loop over i */
								}
							}
						  }
					    }
					}
/*       sort triplets for match quality (.corr) */
       quicksort_con (con0, match0);

/*     pragmatic version: 
       take triplets from the top to the bottom of the sorted list 
       only if none of the points has already been used 
*/
      for (i=0; i<match0; i++){
		  p1 = con0[i].p[0];	if (p1 > -1)	if (++tim[0][p1] > 1)	continue;
		  p2 = con0[i].p[1];	if (p2 > -1)	if (++tim[1][p2] > 1)	continue;
		  p3 = con0[i].p[2];	if (p3 > -1)	if (++tim[2][p3] > 1)	continue;
		  p4 = con0[i].p[3];	if (p4 > -1  && cpar->num_cams > 3) if (++tim[3][p4] > 1) continue;

		  con[match] = con0[i];
		  match++;
	  }

      match3 = match - match4;

      /* repair artifact (?) */
      if (cpar->num_cams == 3) for (i=0; i<match; i++)	con[i].p[3] = -1;
    }
  
/*   search consistent pairs :  12, 13, 14, 23, 24, 34 
      only if an object model is available or if only 2 images are used 
*/
      if(cpar->num_cams > 1 && cpar->allCam_flag == 0){
		  match0 = 0;
		  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
			if ( cpar->num_cams == 2 || (frm->num_targets[0] < 64 && 
			frm->num_targets[1] < 64 && frm->num_targets[2] < 64 && 
			frm->num_targets[3] < 64))
			  for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
			for (i=0; i<frm->num_targets[i1]; i++)
			  {
				p1 = list[i1][i2][i].p1;
				if (p1 > nmax  ||  tim[i1][p1] > 0)	continue;

				/* take only unambigous pairs */
				if (list[i1][i2][i].n != 1)	continue;

				p2 = list[i1][i2][i].p2[0];
				if (p2 > nmax  ||  tim[i2][p2] > 0)	continue;

				corr = list[i1][i2][i].corr[0] / list[i1][i2][i].dist[0];

				if (corr > vpar->corrmin)
				  {
				con0[match0].p[i1] = p1;
				con0[match0].p[i2] = p2;
				con0[match0++].corr = corr;
				  }
			  }

/* 		  sort pairs for match quality (.corr) */
 		  quicksort_con (con0, match0);

/* 		  take pairs from the top to the bottom of the sorted list 
 		  only if none of the points has already been used 
*/
		  for (i=0; i<match0; i++) {
			  p1 = con0[i].p[0];	if (p1 > -1)	if (++tim[0][p1] > 1)	continue;
			  p2 = con0[i].p[1];	if (p2 > -1)	if (++tim[1][p2] > 1)	continue;
			  p3 = con0[i].p[2];	if (p3 > -1  && cpar->num_cams > 2)
			  if (++tim[2][p3] > 1)	continue;
			  p4 = con0[i].p[3];	if (p4 > -1  && cpar->num_cams > 3)
			  if (++tim[3][p4] > 1)	continue;

			  con[match++] = con0[i];
			}
		  } 
 
      match2 = match-match4-match3;
    if(cpar->num_cams == 1){
       printf ( "determined %d points from 2D", match1);
       }
     else{
      printf ("%d quadruplets (red), %d triplets (green) and %d unambigous pairs\n",
	      match4, match3, match2);
     }
/*   give each used pix the correspondence number */
  for (i=0; i<match; i++)
    {
      for (j = 0; j < cpar->num_cams; j++)
        {
         /* Skip cameras without a correspondence obviously. */
         if (con[i].p[j] < 0) continue;

	     p1 = corrected[j][con[i].p[j]].pnr;
	     if (p1 > -1 && p1 < 1202590843)
	      {
	        frm->targets[j][p1].tnr= i;
	      }
	    }
    }

    int count1=0;
	j=0;
	
	for (i=0; i < frm->num_targets[j]; i++)
	  {			      	
	    p1 = frm->targets[j][i].tnr;
	    if (p1 == -1 )
	      {
		   count1++;
	      }
	  }
    printf("unidentified objects = %d\n",count1);

/* Retun values: match counts of each clique size */
    match_counts[0] = match4;
    match_counts[1] = match3;
    match_counts[2] = match2;
    match_counts[3] = match;

/*   free memory for lists of correspondences */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
          free (list[i1][i2]);

   free (con0);

  return con;
}

