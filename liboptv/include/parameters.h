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



typedef struct {
    int  	nlay;
    double  n1;
    double  n2[3];
    double  d[3];
    double  n3;
    int     lut;
} mm_np;

/**
 * Checks deep equality between two mm_np struct instances.
 * Returns 1 for equality, 0 otherwise.
 */
int compare_mm_np(mm_np *mm_np1, mm_np *mm_np2)
{
	/*comparing array-type members*/
	int i;
	for (i=0; i<3; i++)
		if (	mm_np1->n2[i] != mm_np2->n2[i]
			||	mm_np1->d[i]  != mm_np2->d[i] )
			return 0;
	/*comparing remaining basic-type members*/
	if (	mm_np1->nlay != mm_np2->nlay
		||	mm_np1->n1 	 != mm_np2->n1
		||	mm_np1->n3	 != mm_np2->n3
		||	mm_np1->lut	 != mm_np2->lut )
		return 0;
	return 1;
}

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
/*Compares two strings*/
int compare_str(char *s1, char *s2)
{
	int i=0;
	while (s1[i]!='\0' && s2[i]!='\0')
	{
		if (s1[i]!=s2[i])
			return 0;
		i++;
	}
	if (s1[i]!=s2[i]) /* only one of the strings ended with '/0'  */
		return 0;
	return 1;
}
/**
 * Checks deep equality between two compare_control_par struct instances.
 * Returns 1 for equality, 0 otherwise.
 */
int compare_control_par(control_par *c1, control_par *c2)
{
	if ( 	c1->num_cams	!= c2->num_cams
		||	c1->hp_flag		!= c2->hp_flag
		||	c1->allCam_flag	!= c2->allCam_flag
		||	c1->tiff_flag	!= c2->tiff_flag
		||	c1->imx			!= c2->imx
		||	c1->imy			!= c2->imy
		||	c1->pix_x		!= c2->pix_x
		||	c1->pix_y		!= c2->pix_y
		||	c1->chfield		!= c2->chfield
		||	compare_mm_np(c1->mm, c2->mm)==0
		||	compare_str(*(c1->img_base_name), *(c2->img_base_name))==0
		||	compare_str(*(c1->cal_img_base_name), *(c2->cal_img_base_name))==0 )
		return 0;
	return 1;
}

control_par * read_control_par(char *filename);
void free_control_par(control_par *cp);

#endif
