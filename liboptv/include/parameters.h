/* Define classes for handling parameters (storing, reading, writing) so that
   all of libptv has a single point of entry rather than different formats in 
   the same code.
*/

#ifndef PARAMETERS_H
#define PARAMETERS_H

/* Length of basenames. Smaller enough than the buffer lengths used by the
   tracking framebuffer, that suffixes can be added. */
#define SEQ_FNAME_MAX_LEN 240

typedef struct {
    char **img_base_name;
    int first, last;
} sequence_par;

sequence_par* read_sequence_par(char *filename);

typedef struct
{
    double  dacc, dangle, dvxmax, dvxmin;
    double dvymax, dvymin, dvzmax, dvzmin;
    int dsumg, dn, dnx, dny, add;
} track_par;

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
    int pix_x;
    int pix_y;
    int chfield; 
    double  n1;
    double  n2[3];
    double  n3;
    double  d[3];
} control_par;




control_par* read_control_par(char *filename);
void free_control_par(control_par *cp);
int compare_control_par(control_par *c1, control_par *c2);


typedef struct {
    int  	nlay; 
    double  n1;
    double  n2[3];
    double  d[3];
    double  n3;
    int     lut;
} mm_np;
    
mm_np* control_par_to_mm_np(control_par *cp);
int compare_mm_np(mm_np *mm1, mm_np *mm2);

#endif

