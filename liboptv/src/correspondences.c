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
#include <stdlib.h>
#include "correspondences.h"


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

/****************************************************
 *  Memory management                               *
 ****************************************************/

/*  deallocate_target_usage_marks() deallocates all subarrays and the top
    array of target usage marks. Assumes all unallocated subarrays are 
    pointing to NULL so deallocation is simple. This setting is handled by
    the allocation function.
    
    Arguments:
    int** tusage - the usage marks array of arrays.
    int num_cams - 
*/
void deallocate_target_usage_marks(int** tusage, int num_cams) {
    int cam;
    for (cam = 0; cam < num_cams; cam++) {
        free(tusage[cam]); 
    }
    free(tusage);
}

/*  safely_allocate_target_usage_marks() allocates space for per-camera
    arrays marking whether a certain target was used . If some allocation
    failed, it cleans up memory and returns NULL. Allocated arrays are zeroed
    out initially by the C library.
*/
int** safely_allocate_target_usage_marks(int num_cams) {
    int cam, error=0;
    int **tusage;
    
    tusage = (int**) malloc(num_cams * sizeof(int *));
    if (tusage == NULL) return NULL;
    
    for (cam = 0; cam < num_cams; cam++) {
        if (error == 0) {
            tusage[cam] = (int *) calloc(nmax, sizeof(int));
            if (tusage[cam] == NULL) 
                error = 1;
        } else {
            tusage[cam] = NULL; /* So free() can be called without corruption */
        }
    }
    if (error == 0) 
        return tusage;
    
    deallocate_target_usage_marks(tusage, num_cams);
    return NULL;
}

/*  deallocate_adjacency_lists() deallocates all pairwise arrays of posisible
    correspondence between targets. Assumes all unallocated subarrays are 
    pointing to NULL so deallocation is simple. This setting is handled by
    the allocation function. "Adjacency" denotes connection in the conceptual
    targets graph.
    
    Arguments:
    correspond* lists[4][4]  - the array of arrays to clear.
    int num_cams - number of cameras to handle (up to 4).
*/
void deallocate_adjacency_lists(correspond* lists[4][4], int num_cams) {
    int c1, c2;
    
    for (c1 = 0; c1 < num_cams - 1; c1++) {
        for (c2 = c1 + 1; c2 < num_cams; c2++) {
            free(lists[c1][c2]);
        }
    }
}

/*  safely_allocate_adjacency_lists() Allocates and initializes pairwise 
    adjacency lists. If an error occurs, cleans up memory and returns 0, 
    otherwise returns 1.
    
    Arguments:
    correspond* lists[4][4] - an existing array of arrays of pointers,
        where allocations will be stored.
    int num_cams - number of cameras to handle (up to 4).
    int *target_counts - an array holding the number of targets in each camera.
*/
int safely_allocate_adjacency_lists(correspond* lists[4][4], int num_cams, 
    int *target_counts) 
{
    int c1, c2, edge, error=0;

    for (c1 = 0; c1 < num_cams - 1; c1++) {
        for (c2 = c1 + 1; c2 < num_cams; c2++) {
            if (error == 0) {
                lists[c1][c2] = (correspond *) malloc(
                    target_counts[c1] * sizeof(correspond));
                if (lists[c1][c2] == NULL) {
                    error = 1;
                    continue;
                }
                
                for(edge=0; edge < target_counts[c1]; edge++) {
                    lists[c1][c2][edge].n = 0;
                    lists[c1][c2][edge].p1 = 0;
                }
            } else {
                lists[c1][c2] = NULL;
            }
        }
    }
    
    if (error == 0) return 1;
    
    deallocate_adjacency_lists(lists, num_cams);
    return 0;
}


/****************************************************************************/
/*         Optimized clique-finders for fixed numbers of cameras            */
/****************************************************************************/

/*  four_camera_matching() marks candidate cliques found from adjacency lists.
    
    Arguments:
    correspond *list[4][4] - the pairwise adjacency lists.
    int base_target_count - number of turgets in the base camera (first camera)
    double accept_corr - minimal correspondence grade for acceptance.
    n_tupel *scratch - scratch buffer to fill with candidate clique data.
    int scratch_size - size of the scratch space. Upon reaching it, the search
        is terminated and only the candidates found by then are returned.
    
    Returns:
    int, the number of candidate cliques found.
*/
int four_camera_matching(correspond *list[4][4], int base_target_count, 
    double accept_corr, n_tupel *scratch, int scratch_size) 
{
    int i, j, k, l, m, n, o; /* Target counters */
    int p1, p2, p3, p4, p31, p41, p42; /* target pointers */
    int matched = 0;
    double corr;
    
    for (i = 0; i < base_target_count; i++) {
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
            for (n = 0; n < list[1][3][p2].n; n++) {
              p41 = list[1][3][p2].p2[n];
              if (p4 != p41) continue;
              for (o = 0; o < list[2][3][p3].n; o++) {
                  p42 = list[2][3][p3].p2[o];
                  if (p4 != p42) continue;
                  
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
                  
                  if (corr <= accept_corr)
                      continue;

                  /*accept as preliminary match */
                  scratch[matched].p[0] = p1;
                  scratch[matched].p[1] = p2;
                  scratch[matched].p[2] = p3;
                  scratch[matched].p[3] = p4;
                  scratch[matched].corr = corr;
                  
                  matched++;
                  if (matched == scratch_size) {
                      printf ("Overflow in correspondences.");
                      return matched;
                  }
              }
            }
          } /* target loops */
        } /* Other camera loops*/
    } /* 1st camera targets*/
    return matched;
}


/****************************************************************************/
/*--------------- 4 camera model: consistent quadruplets -------------------*/
/****************************************************************************/

n_tupel *correspondences (frame *frm, volume_par *vpar, control_par *cpar, 
  Calibration **calib, int match_counts[]) 
{
  int 	i,j,k,l,m,n,o,i1,i2,i3,cam,part;
  int   count, match=0, match0=0, match4=0, match3=0, match2=0, match1=0;
  int 	p1,p2,p3,p4, p31, p41, p42;
  int  	pt1;
  double       	xa12,ya12,xb12,yb12;
  double       	corr;
  candidate   	cand[MAXCAND];
  n_tupel     	*con0, *con;
  correspond  	*list[4][4];
  coord_2d **corrected;
  int **tim;
  
  /* Allocation of scratch buffers for internal tasks and return-value 
     space. 
  */
  con0 = (n_tupel *) malloc(cpar->num_cams*nmax * sizeof(n_tupel));
  con = (n_tupel *) malloc(cpar->num_cams*nmax * sizeof(n_tupel)); 
  
  tim = safely_allocate_target_usage_marks(cpar->num_cams);
  if (tim == NULL) {
      fprintf(stderr, "out of memory\n");
      free(con0);
      free(con);
      return NULL;
  }
  
  /* allocate memory for lists of correspondences */
  if (safely_allocate_adjacency_lists(list, cpar->num_cams, frm->num_targets) == 0) {
      fprintf(stderr, "list is not allocated");
      deallocate_target_usage_marks(tim, cpar->num_cams);
      free(con0);
      free(con);
      return NULL;
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
      This loop does the correction and also sets point numbers correctly.
      I have a feeling that this should not happen here, because the correction
      is used both for this and for point positions, so the function should 
      receive it as argument.
  */
  
  corrected = (coord_2d **) malloc(cpar->num_cams * sizeof(coord_2d *));
  for (cam = 0; cam < cpar->num_cams; cam++) {
      corrected[cam] = (coord_2d *) malloc(
          frm->num_targets[cam] * sizeof(coord_2d));
      if (corrected[cam] == NULL){
          fprintf(stderr, "corrected is not allocated");
          deallocate_adjacency_lists(list, cpar->num_cams);
          deallocate_target_usage_marks(tim, cpar->num_cams);
          free(con0);
          free(con);
          return NULL;
      }
            
      for (part = 0; part < frm->num_targets[cam]; part++) {
          pixel_to_metric(&corrected[cam][part].x, 
                          &corrected[cam][part].y,
                          frm->targets[cam][part].x, 
                          frm->targets[cam][part].y,
                          cpar);
                            
          dist_to_flat(corrected[cam][part].x, corrected[cam][part].y,
              calib[cam], &corrected[cam][part].x, &corrected[cam][part].y,
              0.00001);
          
          corrected[cam][part].pnr = frm->targets[cam][part].pnr;
      }
        
      /* This is expected by find_candidate_plus() */
      quicksort_coord2d_x(corrected[cam], frm->num_targets[cam]);
  }

  /* Generate adjacency lists: mark candidates for correspondence.
     matching  1 -> 2,3,4  +  2 -> 3,4  +  3 -> 4 */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++) {
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++) {

      for (i=0; i<frm->num_targets[i1]; i++)	if (corrected[i1][i].x != -999) {
          epi_mm (corrected[i1][i].x, corrected[i1][i].y, calib[i1], calib[i2], 
          cpar->mm, vpar, &xa12, &ya12, &xb12, &yb12);
	  
          /* origin point in the list */
	      p1 = i;  list[i1][i2][p1].p1 = p1;	pt1 = corrected[i1][p1].pnr;

          /* search for a conjugate point in corrected[i2] */
          count = find_candidate(corrected[i2], frm->targets[i2], 
              frm->num_targets[i2], xa12, ya12, xb12, yb12, 
              frm->targets[i1][pt1].n, frm->targets[i1][pt1].nx,
              frm->targets[i1][pt1].ny, frm->targets[i1][pt1].sumg, cand, 
              vpar, cpar, calib[i2]);
             
          /* write all corresponding candidates to the preliminary list 
 	         of correspondences */
          if (count > MAXCAND) count = MAXCAND;
	      for (j = 0; j < count; j++) {
	        list[i1][i2][p1].p2[j] = cand[j].pnr;
	        list[i1][i2][p1].corr[j] = cand[j].corr;
	        list[i1][i2][p1].dist[j] = cand[j].tol;
	      }
	      list[i1][i2][p1].n = count;
	  }
    }
  }

  /*   search consistent quadruplets in the list */
  if (cpar->num_cams == 4) {
    match0 = four_camera_matching(list, frm->num_targets[0], 
        vpar->corrmin, con0, 4*nmax);
    
    /* sort quadruplets for match quality (.corr) */
    quicksort_con (con0, match0);

    /*  take quadruplets from the top to the bottom of the sorted list
        only if none of the points has already been used */
    for (i = 0; i < match0; i++) {
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

  /* Image coordinates not needed beyond this point. */
  for (cam = 0; cam < frm->num_cams; cam++) {
      free(corrected[cam]);
  }
  free(corrected);
    
  /* free all other allocations */
  deallocate_adjacency_lists(list, cpar->num_cams);
  deallocate_target_usage_marks(tim, cpar->num_cams);
  free (con0);

  return con;
}
