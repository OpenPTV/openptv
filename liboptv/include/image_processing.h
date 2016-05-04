
/* Forward declarations for various image processing routines */

#ifndef IMAGE_PROCESSING_C
#define IMAGE_PROCESSING_C

#include "parameters.h"

typedef double filter_t[3][3];

int filter_3(unsigned char *img, unsigned char *img_lp, filter_t filt,
    control_par *cpar);
void lowpass_3(unsigned char *img, unsigned char *img_lp, control_par *cpar);
int fast_box_blur(int filt_span, unsigned char *src, unsigned char *dest, 
    control_par *cpar);
void split(unsigned char *img, int half_selector, control_par *cpar);
void subtract_img(unsigned char *img1, unsigned char *img2, unsigned char *img_new, 
    control_par *cpar);
void subtract_mask(unsigned char *img1, unsigned char *img_mask, unsigned char *img_new, 
    control_par *cpar);
void copy_images(unsigned char	*img1, unsigned char *img2, control_par *cpar);
int prepare_image(unsigned char  *img, unsigned char  *img_hp, int dim_lp,
    int filter_hp, char *filter_file, control_par *cpar);

#endif

