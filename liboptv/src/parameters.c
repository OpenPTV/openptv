/* Implementation for parameters handling routines. */

#include "parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>



/* read_sequence_par() reads sequence parameters from a config file with the
   following format: each line is a value, first num_cams values are image 
   names, (num_cams+1)th is the first number in the sequence, (num_cams+2)th 
   line is the last value in the sequence.
   
   Arguments:
   char *filename - path to the text file containing the parameters.
   int num_cams - number of cameras
   
   Returns:
   Pointer to a newly-allocated sequence_par structure. Don't forget to free
   the new memory using free_sequence_par(sp*) function. If reading failed for
   any reason, returns NULL.
*/
sequence_par* read_sequence_par(char *filename, int num_cams) {
    char line[SEQ_FNAME_MAX_LEN];
    FILE* par_file;
    int cam, read_ok;
    sequence_par *ret;
    
    par_file = fopen(filename, "r");
    if (par_file == NULL) {
        return NULL;
    }
    
    /* create new sequence_par struct with memory allocated to all its inner pointers*/
    ret = new_sequence_par(num_cams);

    for (cam = 0; cam < num_cams; cam++) {
        read_ok = fscanf(par_file, "%s\n", line);
        if (read_ok == 0) goto handle_error;
        
        strncpy(ret->img_base_name[cam], line, SEQ_FNAME_MAX_LEN);
    }
    
    if( (read_ok = fscanf(par_file, "%d\n", &(ret->first))) == 0)
        goto handle_error;
    if( (read_ok = fscanf(par_file, "%d\n", &(ret->last))) == 0)    
        goto handle_error;

    fclose(par_file);
    return ret;
    
handle_error:
    printf("Error reading sequence parameters from %s\n", filename);
    free_sequence_par(ret);
    fclose(par_file);
    return NULL;
}

/*  new_sequence_par() creates a new sequence_par struct and allocates memory 
    for its inner pointers.
    
    Arguments:
    int num_cams - number of cameras
    
    Returns:
    Pointer to a newly-allocated sequence_par structure. Don't forget to free
    the new memory using free_sequence_par(sp*) function. If reading failed for
    any reason, returns NULL.
*/
sequence_par * new_sequence_par(int num_cams) {
    int cam;
    sequence_par *ret;

    ret = (sequence_par *) malloc(sizeof(sequence_par));
    ret->img_base_name = (char **) calloc(num_cams, sizeof(char *));

    ret->num_cams = num_cams;
    for (cam = 0; cam < num_cams; cam++) {
        ret->img_base_name[cam] = (char *) malloc(SEQ_FNAME_MAX_LEN);
    }
    return ret;
}

/* compare_sequence_par() checks that all fields of two sequence_par objects are
   equal.

   Arguments:
   sequence_par *sp1, track_par *sp2- addresses of the objects for comparison.

   Returns:
   True if equal, false otherwise. */
int compare_sequence_par(sequence_par *sp1, sequence_par *sp2) {
    int cam;
    
    if (sp1->first != sp2->first || sp1->last != sp2->last
            || sp1->num_cams != sp2->num_cams)
        return 0; /*not equal*/

    for (cam = 0; cam < sp1->num_cams; cam++) {
        if (strcmp(sp1->img_base_name[cam],sp1->img_base_name[cam]) !=0){
            return 0; /*not equal*/
        }
    }
    return 1; /*equal*/
}

/* free_sequence_par() frees the memory allocated for sequence_par struct 
   pointed to by sp and its inner pointers, setting freed pointers to NULL.
   
   Arguments:
   sequence_par *sp - the sequence_par struct to free with the other memory 
      it owns.
*/
void free_sequence_par(sequence_par * sp) {
    int cam;

    for (cam = 0; cam < sp->num_cams; cam++) {
        free(sp->img_base_name[cam]);
        sp->img_base_name[cam] = NULL;
    }
    free(sp->img_base_name);
    sp->img_base_name = NULL;

    free(sp);
    sp = NULL;
}

/* read_track_par() reads tracking parameters from a config file with the
   following format: each line is a value, in this order:
   1. dvxmin
   2. dvxmax
   3. dvymin
   4. dvymax
   5. dvzmin
   6. dvzmax
   7. dangle
   8. dacc
   9. add
   
   Arguments:
   char *filename - path to the text file containing the parameters.
   
   Returns:
   Pointer to a newly-allocated track_par structure. If reading failed for 
   any reason, returns NULL.
*/
track_par* read_track_par(char *filename) {
    FILE* fpp;
    track_par *ret = (track_par *) malloc(sizeof(track_par));
    
/* @WARNING: This is really important, some libraries (e.g. ROS, Qt4) seems to set the 
system locale which takes decimal commata instead of points which causes the file input 
parsing to fail */    
    setlocale(LC_NUMERIC,"C");
    
    fpp = fopen(filename, "r");
    if(fscanf(fpp, "%lf\n", &(ret->dvxmin)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dvxmax)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dvymin)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dvymax)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dvzmin)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dvzmax)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dangle)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->dacc)) == 0) goto handle_error;
    if(fscanf(fpp, "%d\n", &(ret->add)) == 0) goto handle_error;
    fclose (fpp);
    
    ret->dsumg = ret->dn = ret->dnx = ret->dny = 0;
    return ret;

handle_error:
    printf("Error reading tracking parameters from %s\n", filename);
    free(ret);
    fclose (fpp);
    return NULL;
}

/* compare_track_par() checks that all fields of two track_par objects are
   equal.
   
   Arguments:
   track_par *t1, track_par *t2 - addresses of the objects for comparison.
   
   Returns:
   True if equal, false otherwise.
*/
int compare_track_par(track_par *t1, track_par *t2) {
    return ((t1->dvxmin == t2->dvxmin) && (t1->dvxmax == t2->dvxmax) && \
        (t1->dvymin == t2->dvymin) && (t1->dvymax == t2->dvymax) && \
        (t1->dvzmin == t2->dvzmin) && (t1->dvzmax == t2->dvzmax) && \
        (t1->dacc == t2->dacc) && (t1->dangle == t2->dangle) && \
        (t1->dsumg == t2->dsumg) && (t1->dn == t2->dn) && \
        (t1->dnx == t2->dnx) && (t1->dny == t2->dny) && (t1->add == t2->add));
}

/* read_volume_par() reads parameters of illuminated volume from a config file
   with the following format: each line is a value, in this order:
   1.  X_lay[0], (mm) leftmost X boundary 
   2.  Zmin_lay[0], (mm), left size closest Z point 
   3.  Zmax_lay[0], (mm), left side farest Z point
   4.  X_lay[1], (mm) rightmost X boundary
   5.  Zmin_lay[1] (mm), right side, closest Z
   6.  Zmax_lay[1] (mm), right side, farest Z
   7.  cnx, correlation limit for nx size of a candidate blob
   8.  cny, correlation limit for ny size of a candidate blob
   9.  cn, correlation limit for n particle size of a candidate blob
   10. csumg, correlation limit for sum of grey scale of a candidate
   11. corrmin, minimum correlation of all above parameters
   12. eps0 (mm), flat coordinates, see docs/ptv_coordinates
   
   Arguments:
   char *filename - path to the text file containing the parameters.
   
   Returns:
   Pointer to a newly-allocated volume_par structure. If reading failed for 
   any reason, returns NULL.
*/
volume_par* read_volume_par(char *filename) {
    FILE* fpp;
    volume_par *ret = (volume_par *) malloc(sizeof(volume_par));

/* @WARNING: This is really important, some libraries (e.g. ROS, Qt4) seems to set the 
system locale which takes decimal commata instead of points which causes the file input 
parsing to fail */
    setlocale(LC_NUMERIC,"C");
    
    fpp = fopen(filename, "r");
    if(fscanf(fpp, "%lf\n", &(ret->X_lay[0])) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->Zmin_lay[0])) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->Zmax_lay[0])) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->X_lay[1])) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->Zmin_lay[1])) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->Zmax_lay[1])) == 0) goto handle_error;
    
    if(fscanf(fpp, "%lf\n", &(ret->cnx)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->cny)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->cn)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->csumg)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->corrmin)) == 0) goto handle_error;
    if(fscanf(fpp, "%lf\n", &(ret->eps0)) == 0) goto handle_error;
    
    fclose (fpp);
    return ret;

handle_error:
    printf("Error reading volume parameters from %s\n", filename);
    free(ret);
    fclose(fpp);
    return NULL;
}

/* compare_volume_par() checks that all fields of two volume_par objects are
   equal.
   
   Arguments:
   volume_par *v1, volume_par *v2 - addresses of the objects for comparison.
   
   Returns:
   True if equal, false otherwise.
*/
int compare_volume_par(volume_par *v1, volume_par *v2) {
    return ( 
        (v1->X_lay[0] == v2->X_lay[0]) && \
        (v1->Zmin_lay[0] == v2->Zmin_lay[0]) && \
        (v1->Zmax_lay[0] == v2->Zmax_lay[0]) && \
        (v1->X_lay[1] == v2->X_lay[1]) && \
        (v1->Zmin_lay[1] == v2->Zmin_lay[1]) && \
        (v1->Zmax_lay[1] == v2->Zmax_lay[1]) &&
        (v1->cn == v2->cn) && (v1->cnx == v2->cnx) && \
        (v1->cny == v2->cny) && (v1->csumg == v2->csumg) && \
        (v1->corrmin == v2->corrmin) && (v1->eps0 == v2->eps0) );
}

/* new_control_par() allocates memory for a control_par struct and the other 
   memory it owns.
   
   Arguments:
   int cams - number of cameras for whose data we need memory allocation.
   
   Returns:
   Pointer to the newly allocated memory for the control_par struct.
*/
control_par * new_control_par(int cams) {
    int cam;
    control_par *ret = (control_par *) malloc(sizeof(control_par));

    ret->num_cams = cams;

    ret->img_base_name = (char **) calloc(ret->num_cams, sizeof(char*));
    ret->cal_img_base_name = (char **) calloc(ret->num_cams, sizeof(char *));

    for (cam = 0; cam < ret->num_cams; cam++) {
        ret->img_base_name[cam] = (char *) malloc(SEQ_FNAME_MAX_LEN);
        ret->cal_img_base_name[cam] = (char *) malloc(SEQ_FNAME_MAX_LEN);
    }

    ret->mm = (mm_np *) malloc(sizeof(mm_np));

    return ret;
}

/*  read_control_par() reads general control parameters that are not present in
    other config files but are needed generally. The arguments are read in
    this order:
    
    1. num_cams - number of cameras in a frame.
    2n (n = 1..4). img_base_name
    2n + 1. cal_img_base_name
    10 hp_flag - high pass filter flag (0/1)
    11. allCam_flag - flag using the particles that are matched in all cameras
    12. tiff_flag, use TIFF headers or not (if RAW images) 0/1
    13. imx - horizontal size of the image/sensor in pixels, e.g. 1280
    14. imy - vertical size in pixels, e.g. 1024
    15. pix_x
    16 pix_y - pixel size of the sensor (one value per experiment means 
   that all cameras are identical. TODO: allow for different cameras), in [mm], 
   e.g. 0.010 = 10 micron pixel
    17. chfield - whether to use whole image (0), upper half (1) or lower (2).
    18. mmp.n1 - index of refraction of the first media (air = 1)
    19. mmp.n2[0] - index of refraction of the second media - glass windows, 
        can be different?
    20. mmp.n3 - index of refraction of the flowing media (air, liquid)
    21. mmp.d[0] - thickness of the glass/perspex windows (second media), can be
        different ?
    
    (21 lines overall, regardless of camera count)
    
    Arguments:
    char *filename - path to the text file containing the parameters.
    
    Returns:
    Pointer to a newly-allocated control_par structure. If reading failed for 
    any reason, returns NULL.
*/
control_par* read_control_par(char *filename) {
    char line[SEQ_FNAME_MAX_LEN];
    FILE* par_file;
    int cam;
    int num_cams;
    control_par *ret;

/* @WARNING: This is really important, some libraries (e.g. ROS, Qt4) seems to set the 
system locale which takes decimal commata instead of points which causes the file input 
parsing to fail */
    setlocale(LC_NUMERIC,"C");

    if ((par_file = fopen(filename, "r")) == NULL) {
        printf("Could not open file %s", filename);
        return NULL;
    }

    if (fscanf(par_file, "%d\n", &num_cams) != 1) {
        printf("Could not read number of cameras from %s", filename);
        return NULL;
    }
    ret = new_control_par(num_cams);
    for (cam = 0; cam < ret->num_cams; cam++) {
        if (fscanf(par_file, "%s\n", line) == 0) goto handle_error;
        strncpy(ret->img_base_name[cam], line, SEQ_FNAME_MAX_LEN);
        
        if (fscanf(par_file, "%s\n", line) == 0) goto handle_error;
        strncpy(ret->cal_img_base_name[cam], line, SEQ_FNAME_MAX_LEN);
    }
    
    
    if(fscanf(par_file, "%d\n", &(ret->hp_flag)) == 0) goto handle_error;
    if(fscanf(par_file, "%d\n", &(ret->allCam_flag)) == 0) goto handle_error;
    if(fscanf(par_file, "%d\n", &(ret->tiff_flag)) == 0) goto handle_error;
    if(fscanf(par_file, "%d\n", &(ret->imx)) == 0) goto handle_error;
    if(fscanf(par_file, "%d\n", &(ret->imy)) == 0) goto handle_error;
    if(fscanf(par_file, "%lf\n", &(ret->pix_x)) == 0) goto handle_error;
    if(fscanf(par_file, "%lf\n", &(ret->pix_y)) == 0) goto handle_error;
    if(fscanf(par_file, "%d\n", &(ret->chfield)) == 0) goto handle_error;
    if(fscanf(par_file, "%lf\n", &(ret->mm->n1)) == 0) goto handle_error;
    if(fscanf(par_file, "%lf\n", &(ret->mm->n2[0])) == 0) goto handle_error;
    if(fscanf(par_file, "%lf\n", &(ret->mm->n3)) == 0) goto handle_error;
    if(fscanf(par_file, "%lf\n", &(ret->mm->d[0])) == 0) goto handle_error; 
    
    fclose(par_file);
    ret->mm->nlay = 1;
    return ret;

handle_error:
    printf("Error reading control parameters from %s\n", filename);
    fclose(par_file);
    free_control_par(ret);    
    return NULL;
}

/*  free_control_par() frees a control_par pointer and the memory allocated
    under it for image names etc.
    
    Arguments:
    control_par *cp - pointer to the control_par object to destroy.
*/
void free_control_par(control_par *cp) {
    int cam;
    for (cam = 0; cam < cp->num_cams; cam++) {
        free(cp->img_base_name[cam]);
        cp->img_base_name[cam] = NULL;
        free(cp->cal_img_base_name[cam]);
        cp->cal_img_base_name[cam] = NULL;
    }
    free(cp->img_base_name);
    cp->img_base_name = NULL;

    free(cp->cal_img_base_name);
    cp->cal_img_base_name = NULL;

    free(cp->mm);
    cp->mm = NULL;

    free(cp);
    cp = NULL;
}

/* compare_control_par() checks that two control_par objects are deeply-equal,
   i.e. the memorry allocations contain equal values, and other fields are
   directly equal.

   Arguments:
   control_par *v1, *v2 - addresses of the objects for comparison.

   Returns:
   True if equal, false otherwise.
*/
int compare_control_par(control_par *c1, control_par *c2) {
    int cam;

    if (c1->num_cams != c2->num_cams) return 0;

    for (cam = 0; cam < c1->num_cams; cam++) {
        if (strncmp(c1->img_base_name[cam], c2->img_base_name[cam],
            SEQ_FNAME_MAX_LEN - 1) != 0) return 0;
        if (strncmp(c1->cal_img_base_name[cam], c2->cal_img_base_name[cam],
            SEQ_FNAME_MAX_LEN - 1) != 0) return 0;
    }

    if (c1->hp_flag != c2->hp_flag) return 0;
    if (c1->allCam_flag != c2->allCam_flag) return 0;
    if (c1->tiff_flag != c2->tiff_flag) return 0;
    if (c1->imx != c2->imx) return 0;
    if (c1->imy != c2->imy) return 0;
    if (c1->pix_x != c2->pix_x) return 0;
    if (c1->pix_y != c2->pix_y) return 0;
    if (c1->chfield != c2->chfield) return 0;

    if(compare_mm_np(c1->mm, c2->mm)==0) return 0;

    return 1;
}

/* Checks deep equality between two mm_np struct instances.
 * Returns 1 for equality, 0 otherwise.
 * PLEASE NOTE: only first elements in n2 and in d are checked.
 */
int compare_mm_np(mm_np *mm_np1, mm_np *mm_np2)
{
	//comparing first elements only of n2 and d
	if (	mm_np1->n2[0] != mm_np2->n2[0]
		|| 	mm_np1->d[0]  != mm_np2->d[0])
		return 0;
	//comparing primitive type variables
	if (	mm_np1->nlay != mm_np2->nlay
		||	mm_np1->n1 	 != mm_np2->n1
		||	mm_np1->n3	 != mm_np2->n3 )
		return 0;
	return 1;
}

/* Reads target recognition parameters from file.
 * Parameter: filename - the absolute/relative path to file from which the parameters 
              will be read.
 * Returns: pointer to a new target_par structure.
 */
target_par* read_target_par(char *filename) {
    target_par *ret;

    FILE * file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open target recognition parameters file %s.\n", filename);
        return NULL;
    }

    ret = malloc(sizeof(target_par));

    if (   !(fscanf(file, "%d", &ret->gvthres[0])==1)   /* threshold for binarization 1.image */
        || !(fscanf(file, "%d", &ret->gvthres[1])==1)   /* threshold for binarization 2.image */
        || !(fscanf(file, "%d", &ret->gvthres[2])==1)   /* threshold for binarization 3.image */
        || !(fscanf(file, "%d", &ret->gvthres[3])==1)   /* threshold for binarization 4.image */
        || !(fscanf(file, "%d", &ret->discont)==1)      /* max discontinuity */
        || !(fscanf(file, "%d  %d", &ret->nnmin, &ret->nnmax)==2) /* min. and max. number of */
        || !(fscanf(file, "%d  %d", &ret->nxmin, &ret->nxmax)==2) /* pixels per target,  */
        || !(fscanf(file, "%d  %d", &ret->nymin, &ret->nymax)==2) /* abs, in x, in y     */
        || !(fscanf(file, "%d", &ret->sumg_min)==1)               /* min. sumg */
        || !(fscanf(file, "%d", &ret->cr_sz)==1))                 /* size of crosses */
    {
        printf("Error reading target recognition parameters from %s\n", filename);
        free(ret);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return ret;
}

/* Checks deep equality between two target_par structure variables.
 * Returns 1 for equality, 0 otherwise.
 */
int compare_target_par(target_par *targ1, target_par *targ2) {
    return (   targ1->discont ==    targ2->discont
            && targ1->gvthres[0] == targ2->gvthres[0]
            && targ1->gvthres[1] == targ2->gvthres[1]
            && targ1->gvthres[2] == targ2->gvthres[2]
            && targ1->gvthres[3] == targ2->gvthres[3]
            && targ1->nnmin ==      targ2->nnmin
            && targ1->nnmax ==      targ2->nnmax
            && targ1->nxmin ==      targ2->nxmin
            && targ1->nxmax ==      targ2->nxmax
            && targ1->nymin ==      targ2->nymin
            && targ1->nymax ==      targ2->nymax
            && targ1->sumg_min ==   targ2->sumg_min
            && targ1->cr_sz ==      targ2->cr_sz);
}
/* Writes target_par structure contents to a file.
 * Parameters:
 * targ - a pointer to target_par structure that will be written to file
 * filename - pointer to char array representing the absolute/relative file name
 */
void write_target_par(target_par *targ, char *filename) {
    FILE *file = fopen(filename, "w");

    if (file == NULL)
        printf("Can't create file: %s\n", filename);

    fprintf(file, "%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d",
            targ->gvthres[0],
            targ->gvthres[1],
            targ->gvthres[2],
            targ->gvthres[3],
            targ->discont,
            targ->nnmin,
            targ->nnmax,
            targ->nxmin,
            targ->nxmax,
            targ->nymin,
            targ->nymax,
            targ->sumg_min,
            targ->cr_sz);

    fclose(file);
}
