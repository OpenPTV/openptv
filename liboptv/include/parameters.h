/* Define classes for handling parameters (storing, reading, writing) so that
 *  all of libptv has a single point of entry rather than different formats in
 * the same code.
 */
#ifndef PARAMETERS_H
#define PARAMETERS_H

/* Length of basenames. Smaller enough than the buffer lengths used by the
   tracking framebuffer, that suffixes can be added. */
#define SEQ_FNAME_MAX_LEN 240

typedef struct {
    int num_cams;
    char **img_base_name;
    int first, last;
} sequence_par;

sequence_par* read_sequence_par(char *filename, int num_cams);
sequence_par* new_sequence_par(int num_cams);
void free_sequence_par(sequence_par * sp);
int compare_sequence_par(sequence_par *sp1, sequence_par *sp2);

typedef struct {
    double  dacc, dangle, dvxmax, dvxmin;
    double dvymax, dvymin, dvzmax, dvzmin;
    int dsumg, dn, dnx, dny, add;
} track_par;

/* Function declarations */
track_par* read_track_par(char *filename);
int compare_track_par(track_par *t1, track_par *t2);

typedef struct {
    double X_lay[2], Zmin_lay[2], Zmax_lay[2];

    /* Criteria for correspondence are in the same file. For now they'll be
    in the same structure, but TODO: separate them. */
    double cn, cnx, cny, csumg, eps0, corrmin;
} volume_par;

volume_par* read_volume_par(char *filename);
int compare_volume_par(volume_par *v1, volume_par *v2);

typedef struct {
    int  	nlay;
    double  n1;
    double  n2[3];
    double  d[3];
    double  n3;
} mm_np;

/* Parameters that control general aspects in the setup and behaviour of
   various parts of the program, like image basenames etc. */
typedef struct {
    int num_cams;
    char **img_base_name; /* Note the duplication with sequence_par. */
    char **cal_img_base_name;
    int hp_flag;
    int allCam_flag;
    int tiff_flag;
    int imx;
    int imy;
    double pix_x;
    double pix_y;
    int chfield;
    mm_np *mm;
} control_par;

control_par * new_control_par(int cams);
control_par * read_control_par(char *filename);
void free_control_par(control_par *cp);

/* Parameters for target recognition */
typedef struct {
    int discont;
    int gvthres[4];
    int nnmin, nnmax;
    int nxmin, nxmax;
    int nymin, nymax;
    int sumg_min;
    int cr_sz;
} target_par;

/* Reads target recognition parameters from file.
 * Parameter: filename - the absolute/relative path to file from which the parameters will be read.
 * Returns: pointer to a new target_par structure. */
target_par* read_target_par(char *filename);

/* Checks deep equality between two target_par structure variables.
 * Returns 1 for equality, 0 otherwise.*/
int compare_target_par(target_par *targ1, target_par *targ2);

/* Writes target_par structure contents to a file.
 * Parameters:
 * targ - a pointer to target_par structure that will be written to file
 * filename - pointer to char array representing the absolute/relative file name */
void write_target_par(target_par *targ, char *filename) ;

/* Checks deep equality between two mm_np struct instances.
 * Returns 1 for equality, 0 otherwise. */
int compare_mm_np(mm_np *mm_np1, mm_np *mm_np2);

/* Checks deep equality between two compare_control_par struct instances.
 * Returns 1 for equality, 0 otherwise. */
int compare_control_par(control_par *c1, control_par *c2);

#endif
