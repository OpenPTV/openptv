/*******************************************************************************
**
** Title: ptv
**
** Author: Heinrich Stueer
**
** Description: Main modul of track.
** Dies ist eine abgespeckte Version vom Malik und Papantoniou (allerdings mit
** ein paar Anderungen)
** Created: 12.02.1998
** Changes:
**
*******************************************************************************/
/*
Copyright (c) 1990-2011 ETH Zurich

See the file license.txt for copying permission.
*/

/* ----- recent changes -------------------------------------------------------
  ad holten, 12-2012
  replaced:
        if		(filenumber < 10)	sprintf (filein,
"res/rt_is.%1d", filenumber); else if (filenumber < 100)	sprintf (filein,
"res/rt_is.%2d",  filenumber);
        else						sprintf (filein,
"res/rt_is.%3d", filenumber); by sprintf (filein, "res/rt_is.%d", filenumber);

  replaced:
        fp = fopen (filein, "r");
        if (! fp) printf("Can't open ascii file: %s\n", filein);
  by
        fp = fopen_rp (filein);		(fopen_rp prints an error message on
failure) if (!fp) return;
------------------------------------------------------------------------------*/

#include "ptv.h"

void level1(void);
void level2(void);
void level3(void);
void neighbours(float seekx[], float radi[], int nliste[], int *innliste,
                int set);

int seq_track_proc_c(ClientData clientData, Tcl_Interp *interp, int argc,
                     const char **argv) {
  int step, i, k;

  /*Alloc space*/
  for (i = 0; i < 4; i++)
    mega[i] = (P *)calloc(sizeof(P), M);

  for (i = 0; i < 4; i++)
    c4[i] = (corres *)calloc(sizeof(corres), M);

  for (i = 0; i < 4; i++)
    for (k = 0; k < n_img; k++)
      t4[i][k] = (target *)calloc(sizeof(target), M);

  readseqtrackcrit();
  /*load again first data sets*/
  step = seq_first;
  read_ascii_data(step);
  rotate_dataset();
  read_ascii_data(step + 1);
  rotate_dataset();
  read_ascii_data(step + 2);

  for (step = (seq_first + 2); step < seq_last; step++) {
    printf("Processing step: %d\n", step);
    tracking(clientData, interp, argc, argv);
    rotate_dataset();
    write_ascii_data(step - 2);
    read_ascii_data(step + 1);
  }

  /*write last data_sets*/

  tracking(clientData, interp, argc, argv);
  rotate_dataset();
  write_ascii_data(step - 2);
  rotate_dataset();
  write_ascii_data(step - 1);

  for (i = 0; i < 4; i++) {
    free(mega[i]);
    free(c4[i]);
    for (k = 0; k < n_img; k++)
      free(t4[i][k]);
  }
  return TCL_OK;
}

void read_ascii_data(int filenumber) {
  FILE *FILEIN;
  char filein[256];
  int i, j;

  for (i = 0; i < M; i++) {
    /*reset all other variable to default value*/
    mega[3][i].prev = -1;
    mega[3][i].next = -2;
    mega[3][i].prio = 4;
    mega[3][i].inlist = 0;
    mega[3][i].finaldecis = 1000000.0;
    c4[3][i].p[0] = -1;
    c4[3][i].p[1] = -1;
    c4[3][i].p[2] = -1;
    c4[3][i].p[3] = -1;
  }

  // replaced next lines, ad holten 12-2012
  //		if		(filenumber < 10)	sprintf (filein, "res/rt_is.%1d",
  //filenumber); 		else if (filenumber < 100)	sprintf (filein,
  //"res/rt_is.%2d",  filenumber);
  //		else						sprintf (filein, "res/rt_is.%3d",
  //filenumber);

  sprintf(filein, "res/rt_is.%d", filenumber);
  FILEIN = fopen_rp(filein); // replaced fopen(), ad holten 12-2012
  if (!FILEIN)
    return;

  i = 0;
  m[3] = 0;

  fscanf(FILEIN, "%*d\n"); // skip the # of 3D points
  do {
    /*read dataset row by row, x,y,z and correspondences */
    fscanf(FILEIN, "%*d %f %f %f %d %d %d %d\n", &mega[3][i].x[0],
           &mega[3][i].x[1], &mega[3][i].x[2], &c4[3][i].p[0], &c4[3][i].p[1],
           &c4[3][i].p[2], &c4[3][i].p[3]);

    c4[3][i].nr = i;

    for (j = 0; j < POSI; j++) {
      mega[3][i].decis[j] = 0.0;
      mega[3][i].linkdecis[j] = -999;
    }
    i++;
    m[3]++;
  } while (!feof(FILEIN));
  fclose(FILEIN);

  /* read targets of each camera */
  for (i = 0; i < n_img; i++) {
    nt4[3][i] = 0;
    compose_name_plus_nr_str(seq_name[i], "_targets", filenumber, filein);

    FILEIN = fopen_rp(filein);  // replaced fopen(), ad holten 12-2012
    if (FILEIN) {
      fscanf(FILEIN, "%d\n", &nt4[3][i]);
      for (j = 0; j < nt4[3][i]; j++) {
        fscanf(FILEIN, "%4d %lf %lf %d %d %d %d %d\n", &t4[3][i][j].pnr,
               &t4[3][i][j].x, &t4[3][i][j].y, &t4[3][i][j].n, &t4[3][i][j].nx,
               &t4[3][i][j].ny, &t4[3][i][j].sumg, &t4[3][i][j].tnr);
      }
      fclose(FILEIN);
    }
  }
}

/**********************************************************************/
/* Added by Alex, 19.04.10 to read _targets only, for the external API */
void read_targets(int i_img, int filenumber, int *num) {
  FILE *FILEIN;
  int j;
  char filein[256];

  compose_name_plus_nr_str(seq_name[i_img], "_targets", filenumber, filein);
  /* read targets of each camera */
  nt4[3][i_img] = 0;

  FILEIN = fopen_rp(filein);  // replaced fopen(), ad holten 12-2012
  if (FILEIN) {
    fscanf(FILEIN, "%d\n", &nt4[3][i_img]);
    for (j = 0; j < nt4[3][i_img]; j++) {
      fscanf(FILEIN, "%4d %lf %lf %d %d %d %d %d\n", &pix[i_img][j].pnr,
             &pix[i_img][j].x, &pix[i_img][j].y, &pix[i_img][j].n,
             &pix[i_img][j].nx, &pix[i_img][j].ny, &pix[i_img][j].sumg,
             &pix[i_img][j].tnr);
    }
    fclose(FILEIN);
  }
  *num = nt4[3][i_img];
}

/**********************************************************************/
void write_ascii_data(int filenumber) {
  FILE *FILEOUT;
  char fileout[256];
  int i, set, j;

  set = 0;

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10)  sprintf (fileout, "res/ptv_is.%1d",
  //filenumber); 	else if (filenumber < 100) sprintf (fileout, "res/ptv_is.%2d",
  //filenumber); 	else					   sprintf (fileout,
  //"res/ptv_is.%3d",	filenumber);

  /*	printf ("write file: %s\n",fileout); */
  sprintf(fileout, "res/ptv_is.%d", filenumber);
  FILEOUT = fopen(fileout, "w");
  if (!FILEOUT) {
    printf("Can't open ascii file for writing\n");
    return; // added, ad holten, 12-2012
  }
  fprintf(FILEOUT, "%d\n", m[set]);

  for (i = 0; i < m[set]; i++) {
    /* write dataset row by row */
    fprintf(FILEOUT, "%4d %4d %10.3f %10.3f %10.3f\n", mega[set][i].prev,
            mega[set][i].next, mega[set][i].x[0], mega[set][i].x[1],
            mega[set][i].x[2]);
  }
  fclose(FILEOUT);

  /* create/update of new targets- and new rt_is-files */

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10) sprintf (fileout, "res/rt_is.%1d",
  //filenumber); 	else if (filenumber< 100) sprintf (fileout, "res/rt_is.%2d",
  //filenumber); 	else					  sprintf (fileout,
  //"res/rt_is.%3d",  filenumber);

  /*	printf ("write file: %s\n",fileout); */
  sprintf(fileout, "res/rt_is.%d", filenumber);
  FILEOUT = fopen(fileout, "w");
  if (!FILEOUT) {
    printf("Can't open ascii file for writing\n");
    return; // added, ad holten 12-2012
  }
  fprintf(FILEOUT, "%d\n", m[set]);

  for (i = 0; i < m[set]; i++) {
    fprintf(FILEOUT, "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n", i + 1,
            mega[set][i].x[0], mega[set][i].x[1], mega[set][i].x[2],
            c4[set][i].p[0], c4[set][i].p[1], c4[set][i].p[2], c4[set][i].p[3]);
  }
  fclose(FILEOUT);

  /* write targets of each camera */
  for (i = 0; i < n_img; i++) {
    compose_name_plus_nr_str(seq_name[i], "_targets", filenumber, fileout);

    FILEOUT = fopen(fileout, "w");
    if (!FILEOUT) {
      printf("Can't open ascii file: %s\n", fileout);
    } else {
      fprintf(FILEOUT, "%d\n", nt4[set][i]);
      for (j = 0; j < nt4[set][i]; j++) {
        fprintf(FILEOUT, "%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n",
                t4[set][i][j].pnr, t4[set][i][j].x, t4[set][i][j].y,
                t4[set][i][j].n, t4[set][i][j].nx, t4[set][i][j].ny,
                t4[set][i][j].sumg, t4[set][i][j].tnr);
      }
      fclose(FILEOUT);
    }
  }
}

void write_added(int filenumber) {
  FILE *FILEOUT;
  char fileout[256];
  int i, set;

  set = 0;

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10) sprintf (fileout, "res/added.%1d",
  //filenumber); 	else if (filenumber< 100) sprintf (fileout, "res/added.%2d",
  //filenumber); 	else					  sprintf (fileout,
  //"res/added.%3d", filenumber);
  sprintf(fileout, "res/added.%d", filenumber);

  /*	printf ("write file: %s\n",fileout); */
  FILEOUT = fopen(fileout, "w");
  if (!FILEOUT)
    printf("Can't open ascii file for writing\n");
  else {
    fprintf(FILEOUT, "%d\n", m[set]);

    for (i = 0; i < m[set]; i++) {
      /*read dataset row by row*/
      fprintf(FILEOUT, "%4d %4d %10.3f %10.3f %10.3f %d\n", mega[set][i].prev,
              mega[set][i].next, mega[set][i].x[0], mega[set][i].x[1],
              mega[set][i].x[2], mega[set][i].prio);
    }
    fclose(FILEOUT);
  }
}

/**********************************************************************/
void write_addedback(int filenumber) {
  FILE *FILEOUT;
  char fileout[256];
  int i, set;

  set = 0;

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10) sprintf (fileout, "res/added.%1d",
  //filenumber); 	else if (filenumber< 100) sprintf (fileout, "res/added.%2d",
  //filenumber); 	else					  sprintf (fileout,
  //"res/added.%3d", filenumber);
  sprintf(fileout, "res/added.%d", filenumber);

  /*	printf ("write file: %s\n",fileout); */
  FILEOUT = fopen(fileout, "w");
  if (!FILEOUT)
    printf("Can't open ascii file for writing\n");
  else {
    fprintf(FILEOUT, "%d\n", m[set]);

    for (i = 0; i < m[set]; i++) {
      /*read dataset row by row, prev/next order changed because backwards*/
      fprintf(FILEOUT, "%4d %4d %10.3f %10.3f %10.3f %d\n", mega[set][i].prev,
              mega[set][i].next, mega[set][i].x[0], mega[set][i].x[1],
              mega[set][i].x[2], mega[set][i].prio);
    }
    fclose(FILEOUT);
  }
}
/* ************************************************************* */

void read_ascii_datanew(int filenumber) {
  FILE *FILEIN;
  char filein[256];
  int i, j;
  int dumy;
  double fdumy;

  for (i = 0; i < M; i++) {
    /*reset all other variable to default value*/
    mega[3][i].prev = -1;
    mega[3][i].next = -2;
    mega[3][i].prio = 4;
    mega[3][i].inlist = 0;
    mega[3][i].finaldecis = 1000000.0;
    c4[3][i].p[0] = -1;
    c4[3][i].p[1] = -1;
    c4[3][i].p[2] = -1;
    c4[3][i].p[3] = -1;
  }

  // replaced next lines, ad holten 12-2012
  //	if (filenumber < 10)	   sprintf (filein, "res/rt_is.%1d",
  //filenumber); 	else if (filenumber < 100) sprintf (filein, "res/rt_is.%2d",
  //filenumber); 	else					   sprintf (filein,
  //"res/rt_is.%3d", filenumber);

  sprintf(filein, "res/rt_is.%d", filenumber);
  FILEIN = fopen_rp(filein); // replaced fopen(), ad holten 12-2012
  if (!FILEIN)
    return;

  m[3] = 0;

  fscanf(FILEIN, "%d\n", &m[3]);

  for (i = 0; i <= m[3]; i++) {
    /*read dataset row by row, x,y,z and correspondences */
    fscanf(FILEIN, "%d %f %f %f %d %d %d %d\n", &dumy, &mega[3][i].x[0],
           &mega[3][i].x[1], &mega[3][i].x[2], &c4[3][i].p[0], &c4[3][i].p[1],
           &c4[3][i].p[2], &c4[3][i].p[3]);

    c4[3][i].nr = i;
    /*reset other variables to default value*/
    mega[3][i].inlist = 0;
    mega[3][i].finaldecis = 1000000.0;

    for (j = 0; j < POSI; j++) {
      mega[3][i].decis[j] = 0.0;
      mega[3][i].linkdecis[j] = -999;
    }
  }
  fclose(FILEIN);

  /* read ptv_is-file for prev and next info */

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10) sprintf (filein, "res/ptv_is.%1d",
  //filenumber); 	else if (filenumber< 100) sprintf (filein, "res/ptv_is.%2d",
  //filenumber); 	else					  sprintf (filein,
  //"res/ptv_is.%3d",  filenumber);

  sprintf(filein, "res/ptv_is.%3d", filenumber);
  FILEIN = fopen_rp(filein); // replaced fopen(), ad holten 12-2012
  if (!FILEIN)
    return;

  fscanf(FILEIN, "%d\n", &dumy);

  for (i = 0; i <= m[3]; i++) {
    /*read dataset row by row*/
    fscanf(FILEIN, "%4d %4d %lf %lf %lf\n", &mega[3][i].prev, &mega[3][i].next,
           &fdumy, &fdumy, &fdumy);
  }
  fclose(FILEIN);
  /* end of read ptv_is-file for prev and next info */

  /* read added-file for prio info */

  // replaced next lines, ad holten 12-2012
  //	if (filenumber < 10)	   sprintf (filein, "res/added.%1d",
  //filenumber); 	else if (filenumber< 100)  sprintf (filein, "res/added.%2d",
  //filenumber); 	else					   sprintf (filein,
  //"res/added.%3d",  filenumber);

  sprintf(filein, "res/added.%d", filenumber);
  FILEIN = fopen_rp(filein); // replaced fopen(), ad holten 12-2012
  if (!FILEIN)
    return;

  // replaced next code, ad holten 12-2012
  //	fscanf(FILEIN, "%d\n", &dumy);
  //	for(i=0; i<=m[3]; i++)
  //	{
  //		/*read dataset row by row*/
  //		fscanf(FILEIN, "%*4d %4d %lf %lf %lf %d\n",
  //			&dumy, &dumy, &fdumy, &fdumy, &fdumy, &mega[3][i].prio);
  //	}

  fscanf(FILEIN, "%*d\n");
  for (i = 0; i < m[3]; i++) /* read dataset row by row */
    fscanf(FILEIN, "%*d %*d %*f %*f %*f %d\n", &mega[3][i].prio);
  fclose(FILEIN);

  /* read targets of each camera */
  for (i = 0; i < n_img; i++) {
    nt4[3][i] = 0;
    compose_name_plus_nr_str(seq_name[i], "_targets", filenumber, filein);

    FILEIN = fopen_rp(filein); // replaced fopen(), ad holten 12-2012
    if (!FILEIN)
      break;

    fscanf(FILEIN, "%d\n", &nt4[3][i]);
    for (j = 0; j < nt4[3][i]; j++) {
      fscanf(FILEIN, "%4d %lf %lf %d %d %d %d %d\n", &t4[3][i][j].pnr,
             &t4[3][i][j].x, &t4[3][i][j].y, &t4[3][i][j].n, &t4[3][i][j].nx,
             &t4[3][i][j].ny, &t4[3][i][j].sumg, &t4[3][i][j].tnr);
    }
    fclose(FILEIN);
  }
}

/**********************************************************************/
void write_ascii_datanew(int filenumber) {
  FILE *FILEOUT;
  char fileout[256];
  int i, set, j;

  set = 0;

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10) sprintf (fileout, "res/ptv_is.%1d",
  //filenumber); 	else if (filenumber< 100) sprintf (fileout, "res/ptv_is.%2d",
  //filenumber); 	else					  sprintf (fileout,
  //"res/ptv_is.%3d",  filenumber);

  sprintf(fileout, "res/ptv_is.%d", filenumber);
  /*	printf ("write file: %s\n",fileout); */
  FILEOUT = fopen(fileout, "w");
  if (!FILEOUT) {
    printf("Can't open ascii file for writing\n");
    return; // added, ad holten 12-2012
  }
  fprintf(FILEOUT, "%d\n", m[set]);

  for (i = 0; i < m[set]; i++) {
    /* write dataset row by row, prev/next order changed because backwards*/
    fprintf(FILEOUT, "%4d %4d %10.3f %10.3f %10.3f\n", mega[set][i].prev,
            mega[set][i].next, mega[set][i].x[0], mega[set][i].x[1],
            mega[set][i].x[2]);
  }
  fclose(FILEOUT);

  /* update of targets- and rt_is-files */

  // replaced next lines, ad holten 12-2012
  //	if		(filenumber < 10) sprintf (fileout, "res/rt_is.%1d",
  //filenumber); 	else if (filenumber< 100) sprintf (fileout, "res/rt_is.%2d",
  //filenumber); 	else					  sprintf (fileout,
  //"res/rt_is.%3d",  filenumber);

  sprintf(fileout, "res/rt_is.%d", filenumber);
  /*	printf ("write file: %s\n",fileout); */
  FILEOUT = fopen(fileout, "w");
  if (!FILEOUT) {
    printf("Can't open ascii file for writing\n");
    return; // added, ad holten 12-2012
  }
  fprintf(FILEOUT, "%d\n", m[set]);

  for (i = 0; i < m[set]; i++) {
    /* write dataset row by row */
    fprintf(FILEOUT, "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n", i + 1,
            mega[set][i].x[0], mega[set][i].x[1], mega[set][i].x[2],
            c4[set][i].p[0], c4[set][i].p[1], c4[set][i].p[2], c4[set][i].p[3]);
  }
  fclose(FILEOUT);

  /* write targets of each camera */
  for (i = 0; i < n_img; i++) {
    compose_name_plus_nr_str(seq_name[i], "_targets", filenumber, fileout);

    FILEOUT = fopen(fileout, "w");
    if (!FILEOUT) {
      printf("Can't open ascii file: %s\n", fileout);
      break; // added, ad holten 12-2012
    }
    fprintf(FILEOUT, "%d\n", nt4[set][i]);
    for (j = 0; j < nt4[set][i]; j++) {
      fprintf(FILEOUT, "%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n",
              t4[set][i][j].pnr, t4[set][i][j].x, t4[set][i][j].y,
              t4[set][i][j].n, t4[set][i][j].nx, t4[set][i][j].ny,
              t4[set][i][j].sumg, t4[set][i][j].tnr);
    }
    fclose(FILEOUT);
  }
}
/* ************************************************************* */

int tracking(ClientData clientData, Tcl_Interp *interp, int argc,
             const char **argv) {
  level1();
  level2();
  level3();

  return TCL_OK;
}

void level1(void) {
  int i, ii, j, k, l;
  float trad[3];
  int liste[M], inliste, n2liste[POSI], inn2liste, n3liste[POSI], inn3liste;
  float seekx[3], esti, acc;
  int finish, flag;

  /* Define some varibles. This is done every time tracking is called
     (my be not necessary but is not a big problem
     trad is radius of correlation neigbourhood, trad1
     is search radius for next timestep with link*/

  trad[0] = (float)tpar.dvxmax;
  trad[1] = (float)tpar.dvymax;
  trad[2] = (float)tpar.dvzmax;

  /* BEGIN TRACKING*/

  /* First start with highest priority this is if particle has already
  link to previous
  timstep current timstep is t[1]*/
  /*first search for tracks with previous links*/
  /*links named -1 or -2 are no links*/

  for (inliste = 0, i = 0; i < m[1]; i++)
    if (mega[1][i].prev > -1) {
      liste[inliste] = i;
      inliste++;
    }
  /*calculate possible tracks for t2 and t3 and calculate decision criteria*/
  if (inliste > 0) {
    for (i = 0; i < inliste; i++) {
      for (j = 0; j < 3; j++)
        seekx[j] =
            2 * mega[1][liste[i]].x[j] - mega[0][mega[1][liste[i]].prev].x[j];

      /*find neighbours in next timestep t = 2*/
      inn2liste = 0;
      neighbours(seekx, trad, n2liste, &inn2liste, 2);
      /* if no neighour is found no link will be established*/

      /*calculate decision criteria*/
      if (inn2liste > 0) {
        for (k = 0; k < inn2liste; k++) {
          for (j = 0; j < 3; j++)
            seekx[j] = 2 * mega[2][n2liste[k]].x[j] - mega[1][liste[i]].x[j];

          /*find neigbours in next timestep t = 3*/
          inn3liste = 0;
          neighbours(seekx, trad, n3liste, &inn3liste, 3);

          if (inn3liste == 0) {
            /*if no neighour in t3 is found, give decision criteria artifical
            value (100000) accelaration can be considered
            as unbelivible large*/
            mega[1][liste[i]].decis[k] = 1000000.0;
            mega[1][liste[i]].linkdecis[k] = n2liste[k];
          } else {
            for (esti = 1000000.0, l = 0; l < inn3liste; l++) {
              /*calculate for estimates the decision value*/
              for (acc = 0.0, ii = 0; ii < 3; ii++)
                acc += sqr(mega[1][liste[i]].x[ii] -
                           2 * mega[2][n2liste[k]].x[ii] +
                           mega[3][n3liste[l]].x[ii]);

              acc = (float)sqrt(acc);
              if (esti > acc)
                esti = acc;
            } /*for(l....)*/
            mega[1][liste[i]].decis[k] = esti;
            mega[1][liste[i]].linkdecis[k] = n2liste[k];
          } /*else(inn2liste >0*/
        }   /*for (k...)*/
        mega[1][liste[i]].inlist = inn2liste;
        if (inn2liste > 1)
          sort(inn2liste, mega[1][liste[i]].decis, mega[1][liste[i]].linkdecis);
      } /*if(inn1liste > 0)*/

    } /*for(i=0....)*/

    /*establish links by streaming completly through the data*/

    do {
      finish = 0;

      for (i = 0; i < inliste; i++) {
        if (mega[1][liste[i]].next < 0) {
          if (mega[1][liste[i]].inlist > 0) {
            /*in the following is a sorted list of decis assumed*/
            flag = 1;
            j = 0;
            do {
              if (mega[2][mega[1][liste[i]].linkdecis[j]].prev < 0) {
                /*found possible link*/
                mega[1][liste[i]].next = mega[1][liste[i]].linkdecis[j];
                mega[2][mega[1][liste[i]].linkdecis[j]].prev = liste[i];
                mega[2][mega[1][liste[i]].linkdecis[j]].prio = 0;
                mega[1][liste[i]].finaldecis = mega[1][liste[i]].decis[j];
                flag = 0;
              } /*if(p2 == -1)*/

              /*test exiting link if would be better*/
              else if (mega[1][mega[2][mega[1][liste[i]].linkdecis[j]].prev]
                           .finaldecis > mega[1][liste[i]].decis[j]) {
                /*current link is better and reset other link*/
                mega[1][mega[2][mega[1][liste[i]].linkdecis[j]].prev].next = -2;

                mega[1][liste[i]].next = mega[1][liste[i]].linkdecis[j];
                mega[2][mega[1][liste[i]].linkdecis[j]].prev = liste[i];
                mega[2][mega[1][liste[i]].linkdecis[j]].prio = 0;
                mega[1][liste[i]].finaldecis = mega[1][liste[i]].decis[j];
                flag = 0;
                finish = 1;
              }

              else {
                j++; /* if first choice is not possible then try next */
              }
            } while ((j < mega[1][liste[i]].inlist) && flag);
          } /*if(mega[1]....)*/
          else {
            mega[1][liste[i]].next = -1; /*No link could be established*/
          }                              /*else*/
        }                                /*if(mega[1]	.next < 0)*/
      }                                  /*for(i=0....)*/
    } while (finish);
  } /*if(inlist >0)*/

  /*END OF FIRST TRAIL*/
}

/*second if no previous link but in neigbouhood exist previous links*/

void level2(void) {
  int i, ii, j, k, l;
  float trad[3];
  int liste[M], inliste, n1liste[POSI], inn1liste;
  int n2liste[POSI], inn2liste, n3liste[POSI], inn3liste;
  float seekx[3], esti, acc, vel[3];
  int finish, flag, nvel;

  /* Define some varibles. This is done every time tracking is called
     (my be not necessary but is not a big problem*/
  trad[0] = (float)tpar.dvxmax;
  trad[1] = (float)tpar.dvymax;
  trad[2] = (float)tpar.dvzmax;

  /* BEGIN TRACKING*/
  /* Secondly start with second priority this is if particle has already no link
   to
   previous timstep but in Particles in neigbourhoud have. Current timstep is
   t[1]*/

  /*first search for tracks with no previous links ancd no next link*/
  /*links named -1 or -2 are no links*/

  for (inliste = 0, i = 0; i < m[1]; i++)
    if (mega[1][i].next < 0 && mega[1][i].prev < 0) {
      /*check if neighbours wihtin correlation have link*/
      for (j = 0; j < 3; j++)
        seekx[j] = mega[1][i].x[j];
      /* search points in neigbourhood within coorelation lenght*/
      inn1liste = 0;
      neighbours(seekx, trad, n1liste, &inn1liste, 1);
      /*check if neighbours have previous link*/
      /*n1liste must be greater than 1 because neigbours will find the point i
       * itself*/
      if (inn1liste > 1) {
        for (vel[0] = 0.0, vel[1] = 0.0, vel[2] = 0.0, nvel = 0, j = 0;
             j < inn1liste; j++) {
          if (n1liste[j] != i)
            if (mega[1][n1liste[j]].prev > -1) {
              for (l = 0; l < 3; l++)
                vel[l] += mega[1][n1liste[j]].x[l] -
                          mega[0][mega[1][n1liste[j]].prev].x[l];
              nvel++;
            }
        }
        if (nvel > 0) {
          /*intermediate storage of center of position in next frame */
          for (l = 0; l < 3; l++)
            mega[1][i].decis[l] = vel[l] / (float)nvel;
          liste[inliste] = i;
          inliste++;
        }
      }
    }

  /*calculate possible tracks for t2 and t3 and calculate decision criteria*/
  if (inliste > 0) {
    for (i = 0; i < inliste; i++) {
      for (j = 0; j < 3; j++)
        seekx[j] = mega[1][liste[i]].x[j] + mega[1][liste[i]].decis[j];

      /*find neighbours in next timestep t = 2*/
      inn2liste = 0;
      neighbours(seekx, trad, n2liste, &inn2liste, 2);
      /* if no neighour is found no link will be established*/

      /*calculate decision criteria*/
      if (inn2liste > 0) {
        for (k = 0; k < inn2liste; k++) {
          for (j = 0; j < 3; j++)
            seekx[j] = 2 * mega[2][n2liste[k]].x[j] - mega[1][liste[i]].x[j];

          /*find neigbours in next timestep t = 3*/
          inn3liste = 0;
          neighbours(seekx, trad, n3liste, &inn3liste, 3);

          if (inn3liste == 0) {
            /*if no neighour in t3 is found, give decision criteria artifical
              value (100000) accelaration can be considered as unbelivible
              large*/
            mega[1][liste[i]].decis[k] = 1000000.0;
            mega[1][liste[i]].linkdecis[k] = n2liste[k];
          } else {
            for (esti = 1000000.0, l = 0; l < inn3liste; l++) {
              /*calculate for estimates the decision value*/
              for (acc = 0.0, ii = 0; ii < 3; ii++)
                acc += sqr(mega[1][liste[i]].x[ii] -
                           2 * mega[2][n2liste[k]].x[ii] +
                           mega[3][n3liste[l]].x[ii]);

              acc = (float)sqrt(acc);
              if (esti > acc)
                esti = acc;
            } /*for(l....)*/
            mega[1][liste[i]].decis[k] = esti;
            mega[1][liste[i]].linkdecis[k] = n2liste[k];
          } /*else(inn2liste >0*/
        }   /*for (k...)*/
        mega[1][liste[i]].inlist = inn2liste;
        if (inn2liste > 1)
          sort(inn2liste, mega[1][liste[i]].decis, mega[1][liste[i]].linkdecis);

      } /*if(inn1liste > 0)*/
    }   /*for(i=0....)*/
        /*establish links by streaming completly through the data*/
    do {
      finish = 0;
      for (i = 0; i < inliste; i++) {
        if (mega[1][liste[i]].next < 0) {

          if (mega[1][liste[i]].inlist > 0) {
            /*in the following is a sorted list of decis assumed*/
            flag = 1;
            j = 0;
            do {
              if (mega[2][mega[1][liste[i]].linkdecis[j]].prev < 0) {
                /*found possible link*/
                mega[1][liste[i]].next = mega[1][liste[i]].linkdecis[j];
                mega[2][mega[1][liste[i]].linkdecis[j]].prev = liste[i];
                mega[2][mega[1][liste[i]].linkdecis[j]].prio = 1;
                mega[1][liste[i]].finaldecis = mega[1][liste[i]].decis[j];
                flag = 0;
              } /*if(p2 == -1)*/

              /*test exiting link if would be better*/
              else if ((mega[1][mega[2][mega[1][liste[i]].linkdecis[j]].prev]
                            .finaldecis > mega[1][liste[i]].decis[j]) &&
                       (mega[2][mega[1][liste[i]].linkdecis[j]].prio >= 1)) {
                /*current link is better and reset other link*/
                mega[1][mega[2][mega[1][liste[i]].linkdecis[j]].prev].next = -2;

                mega[1][liste[i]].next = mega[1][liste[i]].linkdecis[j];
                mega[2][mega[1][liste[i]].linkdecis[j]].prev = liste[i];
                mega[2][mega[1][liste[i]].linkdecis[j]].prio = 1;
                mega[1][liste[i]].finaldecis = mega[1][liste[i]].decis[j];
                flag = 0;
                finish = 1;
              }

              else {
                j++; /* if first choice is not possible then try next */
              }
            } while ((j < mega[1][liste[i]].inlist) && flag);
          } /*if(mega[1]....)*/
          else {
            mega[1][liste[i]].next = -1; /*No link could be established*/
          }                              /*else*/
        }                                /*if(mega[1]  .next<0)*/
      }                                  /*for(i=0....)*/
    } while (finish);
  } /*if(inlist >0)*/
    /*END OF second TRAIL*/
}

/*Third if no previous link nor in neigbouhood exist previous links*/

void level3(void) {
  int i, ii, j, k, l;
  float trad[3];
  int liste[M], inliste, n2liste[POSI], inn2liste, n3liste[POSI], inn3liste;
  float seekx[3], esti, acc;
  int finish, flag;

  /* Define some varibles. This is done every time tracking is called
     (my be not necessary but is not a big problem*/

  trad[0] = (float)tpar.dvxmax;
  trad[1] = (float)tpar.dvymax;
  trad[2] = (float)tpar.dvzmax;

  /* BEGIN TRACKING*/

  /* Thirdly start with third priority this is if particle has no link to
     previous
     timstep and in Particles in neigbourhoud have not. Current timstep is
     t[1]*/

  /*first search for tracks with no previous links and no next link*/
  /*links named -1 or -2 are no links*/

  for (inliste = 0, i = 0; i < m[1]; i++)
    if (mega[1][i].next < 0 && mega[1][i].prev < 0) {
      liste[inliste] = i;
      inliste++;
    }

  /*calculate possible tracks for t2 and t3 and calculate decision criteria*/
  if (inliste > 0) {
    for (i = 0; i < inliste; i++) {
      for (j = 0; j < 3; j++)
        seekx[j] = mega[1][liste[i]].x[j];

      /*find neighbours in next timestep t = 2*/
      inn2liste = 0;
      neighbours(seekx, trad, n2liste, &inn2liste, 2);
      /* if no neighour is found no link will be established*/

      /*calculate decision criteria*/
      if (inn2liste > 0) {
        for (k = 0; k < inn2liste; k++) {
          for (j = 0; j < 3; j++)
            seekx[j] = 2 * mega[2][n2liste[k]].x[j] - mega[1][liste[i]].x[j];
          inn3liste = 0;
          neighbours(seekx, trad, n3liste, &inn3liste, 3);
          if (inn3liste == 0) {
            /* if no neighour in t3 is found, give decision criteria artifical
               value (100000) accelaration can be considered as unbelivible
               large*/
            mega[1][liste[i]].decis[k] = 1000000.0;
            mega[1][liste[i]].linkdecis[k] = n2liste[k];
          } else {
            for (esti = 1000000.0, l = 0; l < inn3liste; l++) {
              /*calculate estimates the decision value*/
              for (acc = 0.0, ii = 0; ii < 3; ii++)
                acc += sqr(mega[1][liste[i]].x[ii] -
                           2 * mega[2][n2liste[k]].x[ii] +
                           mega[3][n3liste[l]].x[ii]);
              acc = (float)sqrt(acc);
              if (esti > acc)
                esti = acc;
            } /*for(l....)*/
            mega[1][liste[i]].decis[k] = esti;
            mega[1][liste[i]].linkdecis[k] = n2liste[k];
          } /*else(inn2liste >0*/
        }   /*for (k...)*/
        mega[1][liste[i]].inlist = inn2liste;
        if (inn2liste > 1)
          sort(inn2liste, mega[1][liste[i]].decis, mega[1][liste[i]].linkdecis);
      } /*if(inn1liste > 0)*/
    }   /*for(i=0....)*/

    /*establish links by streaming completly through the data*/

    do {
      finish = 0;
      for (i = 0; i < inliste; i++) {
        if (mega[1][liste[i]].next < 0) {
          if (mega[1][liste[i]].inlist > 0) {
            /*in the following is a sorted list of decis assumed*/
            flag = 1;
            j = 0;
            do {
              if (mega[2][mega[1][liste[i]].linkdecis[j]].prev < 0) {
                /*found possible link*/
                mega[1][liste[i]].next = mega[1][liste[i]].linkdecis[j];
                mega[2][mega[1][liste[i]].linkdecis[j]].prev = liste[i];
                mega[2][mega[1][liste[i]].linkdecis[j]].prio = 2;
                mega[1][liste[i]].finaldecis = mega[1][liste[i]].decis[j];
                flag = 0;
              } /*if(p2 == -1)*/

              /*test exiting link if would be better*/
              else if ((mega[1][mega[2][mega[1][liste[i]].linkdecis[j]].prev]
                            .finaldecis > mega[1][liste[i]].decis[j]) &&
                       (mega[2][mega[1][liste[i]].linkdecis[j]].prio >= 2)) {
                /*current link is better and reset other link*/
                mega[1][mega[2][mega[1][liste[i]].linkdecis[j]].prev].next = -2;
                mega[1][liste[i]].next = mega[1][liste[i]].linkdecis[j];
                mega[2][mega[1][liste[i]].linkdecis[j]].prev = liste[i];
                mega[2][mega[1][liste[i]].linkdecis[j]].prio = 2;
                mega[1][liste[i]].finaldecis = mega[1][liste[i]].decis[j];
                flag = 0;
                finish = 1;
              } else {
                j++; /* if first choice is not possible then try next */
              }
            } while ((j < mega[1][liste[i]].inlist) && flag);
          } /*if(mega[1]....)*/
          else {
            mega[1][liste[i]].next = -1; /*No link could be established*/
          }                              /*else*/
        }                                /*if(mega[1]	< 0)*/
      }                                  /*for(i=0....)*/
    } while (finish);
  } /*if(inlist >0)*/
    /*END OF THIRD TRAIL*/
    // printf("Tracking results for current step:\n");
    // for (int i = 0; i < m[1]; i++) {
    //   printf("Particle %d: prev=%d next=%d prio=%d finaldecis=%.3f pos=(%.3f, %.3f, %.3f)\n",
    //        i, mega[1][i].prev, mega[1][i].next, mega[1][i].prio, mega[1][i].finaldecis,
    //        mega[1][i].x[0], mega[1][i].x[1], mega[1][i].x[2]);
    // }
    int n_continue = 0, n_stopped = 0, n_new = 0;
    for (int i = 0; i < m[1]; i++) {
      if (mega[1][i].prev >= 0 && mega[1][i].next >= 0) {
        n_continue++;
      } else if (mega[1][i].prev >= 0 && mega[1][i].next < 0) {
        n_stopped++;
      } else if (mega[1][i].prev < 0 && mega[1][i].next >= 0) {
        n_new++;
      }
    }
    printf("Summary for current step: continued=%d, stopped=%d, new=%d\n",
           n_continue, n_stopped, n_new);
}

/***SORTING ALGORIHTMUS****/

void sort(int n, float a[], int b[]) {
  int flag = 0, i, itemp;
  float ftemp;

  do {
    flag = 0;
    for (i = 0; i < (n - 1); i++)
      if (a[i] > a[i + 1]) {
        ftemp = a[i];
        itemp = b[i];
        a[i] = a[i + 1];
        b[i] = b[i + 1];
        a[i + 1] = ftemp;
        b[i + 1] = itemp;
        flag = 1;
      }
  } while (flag);
}

void rotate_dataset(void) {
  void *tmp;
  void *tmp2;
  int i;

  /*rotate dataset by changeing pointer*/
  tmp = mega[0];
  mega[0] = mega[1];
  mega[1] = mega[2];
  mega[2] = mega[3];
  mega[3] = tmp;

  /*rotate counter*/
  m[0] = m[1];
  m[1] = m[2];
  m[2] = m[3];

  tmp = c4[0];
  c4[0] = c4[1];
  c4[1] = c4[2];
  c4[2] = c4[3];
  c4[3] = tmp;

  for (i = 0; i < 4; i++) {
    tmp2 = t4[0][i];
    t4[0][i] = t4[1][i];
    t4[1][i] = t4[2][i];
    t4[2][i] = t4[3][i];
    t4[3][i] = tmp2;

    nt4[0][i] = nt4[1][i];
    nt4[1][i] = nt4[2][i];
    nt4[2][i] = nt4[3][i];
  }
}

void neighbours(float seekx[], float radi[], int nliste[], int *innliste,
                int set) {
  int i;
  /*search for points in srearch radius. No sorted list is supported,
  although sorted in z would increase speed*/

  for (i = 0; i < m[set]; i++) {
    if (fabs(seekx[0] - mega[set][i].x[0]) < radi[0])
      if (fabs(seekx[1] - mega[set][i].x[1]) < radi[1])
        if (fabs(seekx[2] - mega[set][i].x[2]) < radi[2]) {
          nliste[*innliste] = i;
          (*innliste)++;
          if (*innliste > POSI)
            printf("More Points found than can be supported! Reduce search "
                   "area or increase POSI\n");
        }
  }
}
/**
 * @brief Writes the specified targets to the output destination.
 *
 * This function processes the provided targets and writes them to the
 * appropriate output, which could be a file, stream, or other destination.
 *
 * @param targets A collection or list of targets to be written.
 * @param count The number of targets in the collection.
 * @return int Returns 0 on success, or a negative error code on failure.
 */
void write_targets(int i_img, char *img_name, int num, target *pix) {
  int i;
  char filename[1024];
  FILE *fp1;

  snprintf(filename, sizeof(filename), "%s_targets", img_name);
  fp1 = fopen(filename, "w");
  if (!fp1) {
    printf("Can't open ascii file: %s\n", filename);
    return;
  }
  fprintf(fp1, "%d\n", num);
  for (i = 0; i < num; i++) {
    fprintf(fp1, "%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n",
            pix[i].pnr,
            pix[i].x, pix[i].y, pix[i].n,
            pix[i].nx, pix[i].ny, pix[i].sumg,
            pix[i].tnr);
  }
  fclose(fp1);
}