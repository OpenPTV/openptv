import cython
# from libc.stdlib cimport malloc , free

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

cdef extern from "optv/image_processing.h":
    void lowpass_3  "lowpass_3" (unsigned char *img , unsigned char *img_lp,  int imgsize, int imx)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_lowpass_3(np.ndarray[DTYPE_t, ndim=2] img1 not None, np.ndarray[DTYPE_t, ndim=2] img2 not None, imgsize, imx):
    lowpass_3(<unsigned char *>img1.data, <unsigned char *>img2.data, <np.int> imgsize, <np.int> imx)
 
 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def lowpass_3(np.ndarray[DTYPE_t, ndim=2] img, np.ndarray[DTYPE_t, ndim=2] img_lp):
#     """ lowpass_3(img, img_lp) uses 3 x 3 mean filter to create a low passed image
#     Input: 
#          img = np.array of type np.uint8
#     Output: 
#          img_lp = np.array of the same type
#          
#     both input and output have to be allocated       
#     """
#     cdef int imgsize, imx, imy
#             
#     imx, imy = img.shape[0], img.shape[1]
#     imgsize = imx * imy
#     
#     img = np.ascontiguousarray(img)
#     img_lp = np.ascontiguousarray(img_lp)
#     
#     
#     c_lowpass_3(&img[0,0], &img_lp[0,0], imgsize, imx )
# 
#     return None
#    