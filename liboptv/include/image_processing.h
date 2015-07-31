
/* Forward declarations for various image processing routines */

#ifndef IMAGE_PROCESSING_C
#define IMAGE_PROCESSING_C

#include "parameters.h"

typedef double filter_t[3][3];

int filter_3(unsigned char *img, unsigned char *img_lp, filter_t filt,
    control_par *cpar);
void lowpass_3(unsigned char *img, unsigned char *img_lp, control_par *cpar);

#endif

