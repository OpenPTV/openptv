/* Implementation of calibration methods defined in calibration.h */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tracking_frame_buf.h"
#include "calibration.h"


/* Write exterior and interior orientation, and - if available, parameters for
*   distortion corrections.
*
*   Arguments:
*   Exterior Ex - exterior orientation.
*   Interior I - interior orientation.
*   Glass G - glass parameters.
*   ap_52 addp - optional additional (distortion) parameters. NULL is fine if
*      add_file is NULL.
*   char *filename - path of file to contain interior, exterior and glass
*      orientation data.
*   char *add_file - path of file to contain added (distortions) parameters.
*/

int write_ori (Exterior Ex, Interior interior, Glass glass, ap_52 ap,
        char *filename, char *add_file){
  FILE	*fp;
  int  	i, success = 0;

  fp = fopen (filename, "w");
  if (! fp) {
        printf("Can't open ascii file: %s\n", filename);
        goto finalize;
  }
    
  fprintf (fp, "%11.8f %11.8f %11.8f\n    %10.8f  %10.8f  %10.8f\n\n",
	   Ex.x0, Ex.y0, Ex.z0, Ex.omega, Ex.phi, Ex.kappa);
  for (i=0; i<3; i++)  fprintf (fp, "    %10.7f %10.7f %10.7f\n",
				Ex.dm[i][0], Ex.dm[i][1], Ex.dm[i][2]);
  fprintf (fp,"\n    %8.4f %8.4f\n    %8.4f\n", interior.xh, interior.yh, interior.cc);
  fprintf (fp,"\n    %20.15f %20.15f  %20.15f\n", glass.vec_x, glass.vec_y, glass.vec_z);
  
  fclose (fp);
  fp = NULL;
  
  if (add_file == NULL) goto finalize;
  fp = fopen (add_file, "w");
  if (! fp) {
        printf("Can't open ascii file: %s\n", add_file);
        goto finalize;
  }
  fprintf (fp, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f", ap.k1, ap.k2, ap.k3, ap.p1, ap.p2,
    ap.scx, ap.she);
  success = 1;
  
finalize:
    if (fp != NULL) fclose (fp);
    return success;
}

/*  read exterior and interior orientation, and - if available, parameters for
*   distortion corrections.
*   
*   Arguments:
*   Exterior *Ex - output buffer for exterior orientation.
*   Interior *I - output buffer for interior orientation.
*   Glass *G - output buffer for glass parameters.
*   char *ori_file - path of file contatining interior and exterior orientation
*       data
*   ap_52 addp - output buffer for additional (distortion) parameters.
*   char *add_file - path of file contatining added (distortions) parameters.
*   char *add_fallback - path to file for use if add_file can't be openned.
*   
*   Returns:
*   true value on success, false on failure. Failure can happen if add_file
*   can't be opened, or the fscanf results are wrong, but if the additional
*   parameters' file or fallback can't be opened, they're just assigned default
*   values.
*/

int read_ori(Exterior Ex[], Interior I[], Glass G[], char *ori_file,
    ap_52 addp[], char *add_file, char *add_fallback)
{
    FILE	*fp;
    int  	i, scan_res;

    fp = fopen(ori_file, "r");
    if (!fp) {
        printf("Can't open ORI file: %s\n", ori_file);
        goto handle_error;
    }

  /* Exterior */
  scan_res = fscanf (fp, "%lf %lf %lf %lf %lf %lf",
	  &(Ex->x0), &(Ex->y0), &(Ex->z0),
	  &(Ex->omega), &(Ex->phi), &(Ex->kappa));
  if (scan_res != 6) return 0;
  
  /* Exterior rotation matrix */
  for (i=0; i<3; i++) {
    scan_res = fscanf (fp, " %lf %lf %lf",
        &(Ex->dm[i][0]), &(Ex->dm[i][1]), &(Ex->dm[i][2]));
    if (scan_res != 3) return 0;
  }

  /* Interior */
  scan_res = fscanf (fp, "%lf %lf %lf", &(I->xh), &(I->yh), &(I->cc));
  if (scan_res != 3) return 0;
  
  /* Glass */
  scan_res = fscanf (fp, "%lf %lf %lf", &(G->vec_x), &(G->vec_y), &(G->vec_z));
  if (scan_res != 3) return 0;
  
  fclose(fp);
  
  /* Additional: */
  fp = fopen(add_file, "r");
  if ((fp == NULL) && add_fallback) fp = fopen (add_fallback, "r");
  
  if (fp) {
    scan_res = fscanf (fp, "%lf %lf %lf %lf %lf %lf %lf",
        &(addp->k1), &(addp->k2), &(addp->k3), &(addp->p1), &(addp->p2),
        &(addp->scx), &(addp->she));
    fclose (fp);
  } else {
    printf("no addpar fallback used\n"); // Waits for proper logging.
    addp->k1 = addp->k2 = addp->k3 = addp->p1 = addp->p2 = addp->she = 0.0;
    addp->scx=1.0;
  }
  
  return 1;
  
 handle_error:
    if (fp != NULL) fclose (fp);
    return 0;

}



/************************************************
 * compare_exterior() performs a comparison of two Exterior objects.
 *
 * Arguments: 
 * Exterior *e1, *e2 - pointers for the exterior objects to compare.
 *
 * Returns:
 * 1 if equal, 0 if any field differs.
 */
int compare_exterior(Exterior *e1, Exterior *e2) {
    int row, col;
    
    for (row = 0; row < 3; row++)
        for (col = 0; col < 3; col++)
            if (e1->dm[row][col] != e2->dm[row][col])
                return 0;
    
    return ((e1->x0 == e2->x0) && (e1->y0 == e2->y0) && (e1->z0 == e2->z0) \
        && (e1->omega == e2->omega) && (e1->phi == e2->phi) \
        && (e1->kappa == e2->kappa));
}

/************************************************
 * compare_interior() performs a comparison of two Interior objects.
 *
 * Arguments: 
 * Interior *i1, *i2 - pointers for the interior objects to compare.
 *
 * Returns:
 * 1 if equal, 0 if any field differs.
 */
int compare_interior(Interior *i1, Interior *i2) {
    return ((i1->xh == i2->xh) && (i1->yh == i2->yh) && (i1->cc == i2->cc));
}

/************************************************
 * compare_glass() performs a comparison of two Glass objects.
 *
 * Arguments: 
 * Glass *g1, *g2 - pointers for the Glass objects to compare.
 *
 * Returns:
 * 1 if equal, 0 if any field differs.
 */
int compare_glass(Glass *g1, Glass *g2) {
    return ((g1->vec_x == g2->vec_x) && (g1->vec_y == g2->vec_y) && \
        (g1->vec_z == g2->vec_z));
}

/************************************************
 * compare_addpar() performs a comparison of two ap_52 objects.
 *
 * Arguments: 
 * ap_52 *a1, *a2 - pointers for the ap_52 objects to compare.
 *
 * Returns:
 * 1 if equal, 0 if any field differs.
 */
int compare_addpar(ap_52 *a1, ap_52 *a2) {
    return ((a1->k1 == a2->k1) && (a1->k2 == a2->k2) && (a1->k3 == a2->k3) && \
        (a1->p1 == a2->p1) && (a1->p2 == a2->p2) && (a1->scx == a2->scx) && \
        (a1->she == a2->she));
}

/************************************************
 * compare_calib() performs a deep comparison of two Calibration objects.
 *
 * Arguments: 
 * Calibration *c1, *c2 - pointers for the calibration objects to compare.
 *
 * Returns:
 * 1 if equal, 0 if any field differs.
 */
int compare_calib(Calibration *c1, Calibration *c2) {
    return (compare_exterior(&(c1->ext_par), &(c2->ext_par)) && \
        compare_interior(&(c1->int_par), &(c2->int_par)) && \
        compare_glass(&(c1->glass_par), &(c2->glass_par)) && \
        compare_addpar(&(c1->added_par), &(c2->added_par)));
}

/**************************************************
 * read_calibration() reads orientation files and creates a Calibration object
 * that represents the files' data.
 * 
 * Note: for now it uses read_ori(). This is scheduled to change soon.
 * 
 * Arguments:
 * char *ori_file - name of the file containing interior, exterior, and glass
 *   parameters.
 * char *add_file - name of the file containing distortion parameters.
 * char *fallback_file - name of file to use if add_file can't be opened.
 *
 * Returns:
 * On success, a pointer to a new Calibration object. On failure, NULL.
 */
Calibration *read_calibration(char *ori_file, char *add_file,
    char *fallback_file)
{
    Calibration *ret = (Calibration *) malloc(sizeof(Calibration));
    /* indicate that data is not set yet */
    ret->mmlut.data=NULL;

    if (read_ori(&(ret->ext_par), &(ret->int_par), &(ret->glass_par), ori_file,
        &(ret->added_par), add_file, fallback_file))
    {
        rotation_matrix(&(ret->ext_par));
        return ret;
    } else {
        free(ret);
        return NULL;
    }
}


/**************************************************
 * write_calibration() writes to orientation files the data in a Calibration 
 * object.
 * 
 * Note: for now it uses write_ori(). This is scheduled to change soon.
 * 
 * Arguments:
 * Calibration *cal - the calibration data to write.
 * char *ori_file - name of the file to hold interior, exterior, and glass
 *   parameters.
 * char *add_file - name of the file to hold distortion parameters.
 *
 * Returns:
 * True value on success, False otherwise.
 */
int write_calibration(Calibration *cal, char *ori_file, char *add_file) {
    return write_ori(cal->ext_par, cal->int_par, cal->glass_par, 
        cal->added_par, ori_file, add_file);
}

/* rotation_matrix() rotates the Dmatrix of Exterior Ex using
*  three angles of the camera
*
*  Arguments:
*   Exterior Ex
*
*  Returns:
*   modified Exterior Ex
*
*/
 void rotation_matrix (Exterior *Ex) {
 
 double cp,sp,co,so,ck,sk;
 
 
    cp = cos(Ex->phi);
    sp = sin(Ex->phi);
    co = cos(Ex->omega);
    so = sin(Ex->omega);
    ck = cos(Ex->kappa);
    sk = sin(Ex->kappa);

    Ex->dm[0][0] = cp * ck;
    Ex->dm[0][1] = -cp * sk;
    Ex->dm[0][2] = sp;
    Ex->dm[1][0] = co * sk + so * sp * ck;
    Ex->dm[1][1] = co * ck - so * sp * sk;
    Ex->dm[1][2] = -so * cp;
    Ex->dm[2][0] = so * sk - co * sp * ck;
    Ex->dm[2][1] = so * ck + co* sp * sk;
    Ex->dm[2][2] = co * cp;
}


