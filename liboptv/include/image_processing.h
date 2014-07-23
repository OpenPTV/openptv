/* Forward declarations for various image processing routines collected from
image_processing.c, segmentation.c and peakfitting.c */

#ifndef IMAGE_PROCESSING_C
#define IMAGE_PROCESSING_C

#include "calibration.h"
#include "tracking_frame_buf.h"
#include "parameters.h"
#include "lsqadj.h"
#include "ray_tracing.h"
#include "multimed.h"
#include "epi.h"

/* I don't know why this has to be "unsigned" but I have no time to test
   dropping it */
void copy_images (unsigned char	*img1, unsigned char *img2, int imgsize);

void filter_3 (unsigned char *img, unsigned char *img_lp, int imgsize, int imx);
void enhance (unsigned char	*img, int imgsize, int imx );
void histeq (unsigned char	*img, int imgsize, int imx );
void histogram (unsigned char *img, int *hist, int imgsize);
void lowpass_3 (unsigned char *img, unsigned char *img_lp, \
               int imgsize, int imx);
void lowpass_3_cb (unsigned char *img, unsigned char *img_lp, \
               int imgsize, int imx);
void lowpass_n (int n, unsigned char *img, unsigned char *img_lp,\
                int imgsize, int imx);
void unsharp_mask (int n, unsigned char *img0, unsigned char *img_lp, \
                    int imgsize, int imx, int imy);

void zoom (unsigned char *img, unsigned char *zoomimg, int xm, int ym, int zf, \
int imgsize, int imx, int imy);
void zoom_new (unsigned char	*img, unsigned char	*zoomimg, int xm, int ym, int zf,\
int zimx, int zimy, int imx);
void subtract_mask (unsigned char	* img, unsigned char	* img_mask, \
unsigned char	* img_new, int imgsize);

void subtract_img (unsigned char *img1,unsigned char *img2,unsigned char *img_new, int imgsize);
void highpass (unsigned char *img, unsigned char *img_hp, int dim_lp, int filter_hp, int imgsize, int imx);
void handle_imageborders(unsigned char	*img1, unsigned char *img2, int imgsize, int imx);
#endif

