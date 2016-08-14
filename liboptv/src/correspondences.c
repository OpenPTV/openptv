/****************************************************************************

Routine:	       	correspondences.c

Author/Copyright:      	Hans-Gerd Maas

Address:	      	Institute of Geodesy and Photogrammetry
	      		ETH - Hoenggerberg
	      		CH - 8093 Zurich

Creation Date:	       	1988/89

Description:	       	establishment of correspondences for 2/3/4 cameras

****************************************************************************/

#include <optv/epi.h>
#include "correspondences.h"
#include "tools.h"

/* quicksort for list of correspondences in order of match quality */
/* 4 camera version */


void qs_con (con, left, right)
n_tupel	*con;
int    	left, right;
{
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

  if (left < j)	qs_con (con, left, j);
  if (i < right)	qs_con (con, i, right);
}

void quicksort_con (con, num)
n_tupel	*con;
int    	num;
{
  qs_con (con, 0, num-1);
}

/* quicksort of targets in y-order */

void qs_target_y (target *pix, int left, int right) {
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

void quicksort_target_y (target *pix, int num) {
  qs_target_y (pix, 0, num-1);
}


/* quicksort of 2d coordinates in x-order */

void qs_coord2d_x (crd, left, right)
coord_2d	*crd;
int			left, right;
{
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

void quicksort_coord2d_x (coord_2d *crd, int num) {
	qs_coord2d_x (crd, 0, num-1);
}

/****************************************************************************/
/*--------------- 4 camera model: consistent quadruplets -------------------*/
/****************************************************************************/

int correspondences_4 (target pix[][nmax], coord_2d geo[][nmax], int num[], 
    volume_par *vpar, control_par *cpar, Calibration cals[], n_tupel *con, 
    int match_counts[])
{
  int 	i,j,k,l,m,n,o,  i1,i2,i3;
  int   count, match, match0=0, match4=0, match3=0, match2=0, match1=0;
  int 	p1,p2,p3,p4, p31, p41, p42;
  int  	pt1;
  int 	tim[4][nmax];
  double       	xa12,ya12,xb12,yb12,X,Y,Z;
  double       	corr;
  candidate   	cand[maxcand];
  n_tupel     	*con0;
  correspond  	*list[4][4];
  vec3d out;
  FILE *fp1;

  for (j=0; j<4; j++) for (i=0; i<nmax; i++) tim[j][i] = 0;

  /* allocate memory for lists of correspondences */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
    list[i1][i2] = (correspond *) malloc (num[i1] * sizeof (correspond));

  con0 = (n_tupel *) malloc (4*nmax * sizeof (n_tupel));

  /*  initialize ...  */
  match=0; match0=0; match2=0;

  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
      for (i=0; i<num[i1]; i++)
	{
	  list[i1][i2][i].p1 = 0;
	  list[i1][i2][i].n = 0;
	}
  for (i = 0; i < nmax; i++) {
    for (j = 0; j < 4; j++) {
        tim[j][i] = 0;
        con0[i].p[j] = -1;
    }
    con0[i].corr = 0;
  }

  /* -------------if only one cam and 2D--------- */ //by Beat L�thi June 2007
/*
  if(cpar->num_cams == 1){
	  if(res_name[0]==0){
          sprintf (res_name, "rt_is");
	  }
	 fp1 = fopen (res_name, "w");
		fprintf (fp1, "%4d\n", num[0]);
	  for (i=0; i<num[0]; i++){
          epi_mm_2D (geo[0][i].x,geo[0][i].y, &(cals[0]), 
            cpar->mm, vpar, out);
          pix[0][geo[0][i].pnr].tnr=i;
		  fprintf (fp1, "%4d", i+1);
		  fprintf (fp1, " %9.3f %9.3f %9.3f", X, Y, Z);
          fprintf (fp1, " %4d", geo[0][i].pnr);
          fprintf (fp1, " %4d", -1);
          fprintf (fp1, " %4d", -1);
          fprintf (fp1, " %4d\n", -1);
	  }
	  fclose (fp1);
	  match1=num[0];
  }
*/
  /* -------------end of only one cam and 2D ------------ */

  /* matching  1 -> 2,3,4  +  2 -> 3,4  +  3 -> 4 */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++) {
     //printf ("Establishing correspondences  %d - %d\n", i1, i2);
     /* establish correspondences from num[i1] points of img[i1] to img[i2] */

      for (i=0; i<num[i1]; i++)	if (geo[i1][i].x != -999) {
          epi_mm (geo[i1][i].x, geo[i1][i].y, &(cals[i1]), &(cals[i2]), 
          cpar->mm, vpar, &xa12, &ya12, &xb12, &yb12);
	  
    /////ich glaube, da muss ich einsteigen, wenn alles erledigt ist.
	  ///////mit bild_1 x,y Epipole machen und dann selber was schreiben um die Distanz zu messen.
	  ///////zu Punkt in bild_2.

	  /* origin point in the list */
	  p1 = i;  list[i1][i2][p1].p1 = p1;	pt1 = geo[i1][p1].pnr;

	  /* search for a conjugate point in geo[i2] */
      count = find_candidate (geo[i2], pix[i2], num[i2],
            xa12, ya12, xb12, yb12, 
            pix[i1][pt1].n,pix[i1][pt1].nx,pix[i1][pt1].ny,
            pix[i1][pt1].sumg, cand, vpar, cpar, &(cals[i2]) );
	  /* write all corresponding candidates to the preliminary list */
	  /* of correspondences */
	  if (count > maxcand)	{ count = maxcand; }
	  for (j=0; j<count; j++)
	    {
	      list[i1][i2][p1].p2[j] = cand[j].pnr;
	      list[i1][i2][p1].corr[j] = cand[j].corr;
	      list[i1][i2][p1].dist[j] = cand[j].tol;
	    }
	  list[i1][i2][p1].n = count;
	}
  }

  /* ------------------------------------------------------------------ */
  /* search consistent quadruplets in the list */
  if (cpar->num_cams == 4) {
      for (i=0, match0=0; i<num[0]; i++)
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
				      /* accept as preliminary match */
				      con0[match0].p[0] = p1;
				      con0[match0].p[1] = p2;
				      con0[match0].p[2] = p3;
				      con0[match0].p[3] = p4;
				      con0[match0++].corr = corr;
				      if (match0 == 4*nmax)	/* security */
					{
					  printf ("Overflow in correspondences:");
					  printf (" > %d matches\n", match0);
					  i = num[0];
					}
				    }
				}
			    }
              
		      }
            }
		}
	}


      /* -------------------------------------------------------------------- */

      /* sort quadruplets for match quality (.corr) */
      quicksort_con (con0, match0);

      /* -------------------------------------------------------------------- */

      /* take quadruplets from the top to the bottom of the sorted list */
      /* only if none of the points has already been used */
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

  /* ----------------------------------------------------------------------- */
  /* ----------------------------------------------------------------------- */

  /* search consistent triplets :  123, 124, 134, 234 */
  if ((cpar->num_cams == 4 && cpar->allCam_flag == 0) || cpar->num_cams == 3)
    {
      //printf("Search consistent triplets \n");
      match0 = 0;
      for (i1 = 0; i1 < cpar->num_cams - 2; i1++)
        for (i2 = i1 + 1; i2 < cpar->num_cams - 1; i2++)
           for (i3 = i2 + 1; i3 < cpar->num_cams; i3++)
				for (i=0; i<num[i1]; i++){
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
									i = num[i1]; /* Break out of the outer loop over i */
								}
							}
						  }
					    }
					}

      /* ----------------------------------------------------------------------- */

      /* sort triplets for match quality (.corr) */
      quicksort_con (con0, match0);

      /* ----------------------------------------------------------------------- */

      /* pragmatic version: */
      /* take triplets from the top to the bottom of the sorted list */
      /* only if none of the points has already been used */
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

  /* ----------------------------------------------------------------------- */
  /* ----------------------------------------------------------------------- */

  /* search consistent pairs :  12, 13, 14, 23, 24, 34 */
  /* only if an object model is available or if only 2 images are used */
      if(cpar->num_cams > 1 && cpar->allCam_flag == 0){
		  match0 = 0;
		  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
			if ( cpar->num_cams == 2 || (num[0] < 64 && num[1] < 64 && num[2] < 64 && num[3] < 64))
			  for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
			for (i=0; i<num[i1]; i++)
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

		  /* ----------------------------------------------------------------------- */


		  /* sort pairs for match quality (.corr) */
		  quicksort_con (con0, match0);

		  /* ----------------------------------------------------------------------- */


		  /* take pairs from the top to the bottom of the sorted list */
		  /* only if none of the points has already been used */
		  for (i=0; i<match0; i++) {
			  p1 = con0[i].p[0];	if (p1 > -1)	if (++tim[0][p1] > 1)	continue;
			  p2 = con0[i].p[1];	if (p2 > -1)	if (++tim[1][p2] > 1)	continue;
			  p3 = con0[i].p[2];	if (p3 > -1  && cpar->num_cams > 2)
			  if (++tim[2][p3] > 1)	continue;
			  p4 = con0[i].p[3];	if (p4 > -1  && cpar->num_cams > 3)
			  if (++tim[3][p4] > 1)	continue;

			  con[match++] = con0[i];
			}
		  } //end pairs?

     match2 = match-match4-match3;
    if(cpar->num_cams == 1){
       printf ( "determined %d points from 2D", match1);
       }
     else{
      printf ("%d quadruplets (red), %d triplets (green) and %d unambigous pairs\n",
	      match4, match3, match2);
     }
  /* ----------------------------------------------------------------------- */

  /* give each used pix the correspondence number */
  for (i=0; i<match; i++)
    {
      for (j = 0; j < cpar->num_cams; j++)
	{
      /* Skip cameras without a correspondence obviously. */
      if (con[i].p[j] < 0) continue;

	  p1 = geo[j][con[i].p[j]].pnr;
	  if (p1 > -1 && p1 < 1202590843)
	    {
	      pix[j][p1].tnr= i;
	    }
	}
    }
  /* draw crosses on canvas */
    int count1=0;
	j=0;
	for (i=0; i<num[j]; i++)
	  {			      	
	    p1 = pix[j][i].tnr;
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

  /* ----------------------------------------------------------------------- */
  /* free memory for lists of correspondences */
  for (i1 = 0; i1 < cpar->num_cams - 1; i1++)
    for (i2 = i1 + 1; i2 < cpar->num_cams; i2++)
        free (list[i1][i2]);

  free (con0);

  return match;
}

