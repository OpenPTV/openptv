from __future__ import division
import cython
# from libc.stdlib cimport malloc , free

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

cdef extern from "optv/image_processing.h":
    void lowpass_3  (unsigned char *img , unsigned char *img_lp,  int imgsize, int imx)
    void lowpass_3_cb  (unsigned char *img , unsigned char *img_lp,  int imgsize, int imx)
    void lowpass_n (int n, unsigned char *img, unsigned char *img_lp, int imgsize, int imx)
    void copy_images (unsigned char	*img1, unsigned char *img2, int imgsize)
    #void filter_3 "filter_3" (unsigned char *img, unsigned char *img_lp, int imgsize, int imx)
    void filter_3 (unsigned char*, unsigned char*, int, int)
    void highpass (unsigned char *img, unsigned char *img_hp, int dim_lp, int filter_hp, int imgsize, int imx)
    void enhance (unsigned char	*img, int imgsize, int imx )
    void histeq (unsigned char	*img, int imgsize, int imx )


# @cython.boundscheck(False)
# @cython.wraparound(False)
def py_lowpass_3(np.ndarray[DTYPE_t, ndim=2] img not None):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_lp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    lowpass_3(<unsigned char *>img.data, <unsigned char *>img_lp.data, img.shape[0]*img.shape[1], img.shape[1])
    return img_lp

def py_lowpass_3_cb(np.ndarray[DTYPE_t, ndim=2] img not None):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_lp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    lowpass_3_cb(<unsigned char *>img.data, <unsigned char *>img_lp.data, img.shape[0]*img.shape[1], img.shape[1])
    return img_lp
 
# @cython.boundscheck(False)
# @cython.wraparound(False)
def py_lowpass_n(n, np.ndarray[DTYPE_t, ndim=2] img not None):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_lp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    lowpass_n(<int>n, <unsigned char *>img.data, <unsigned char *>img_lp.data, img.shape[0]*img.shape[1], img.shape[1])
    return img_lp

def py_copy_images(np.ndarray[DTYPE_t, ndim=2] img not None):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_copy = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    copy_images(<unsigned char *>img.data, <unsigned char *>img_copy.data, img.shape[0]*img.shape[1])
    return img_copy

def py_filter_3(np.ndarray[unsigned char, ndim=2, mode="c"] img):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_lp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    filter_3(<unsigned char *>img.data, <unsigned char *>img_lp.data, img.shape[0]*img.shape[1], img.shape[1])
    return img_lp
    
def py_highpass(np.ndarray[unsigned char, ndim=2, mode="c"] img, dim_lp, filter_hp):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_hp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    highpass(<unsigned char *>img.data, <unsigned char *>img_hp.data, <int>dim_lp, <int>filter_hp, img.shape[0]*img.shape[1], img.shape[1])
    return img_hp
    
def py_enhance(np.ndarray[unsigned char, ndim=2, mode="c"] img):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_hp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    img_hp = py_copy_images(img)
    enhance(<unsigned char *>img_hp.data, img.shape[0]*img.shape[1], img.shape[1])
    return img_hp

def py_histeq(np.ndarray[unsigned char, ndim=2, mode="c"] img):
    cdef np.ndarray[unsigned char, ndim=2,mode="c"] img_hp = np.empty((img.shape[0],img.shape[1]),dtype='uint8')
    img_hp = py_copy_images(img)
    histeq(<unsigned char *>img_hp.data, img.shape[0]*img.shape[1], img.shape[1])
    return img_hp

# def py_filter_3(np.ndarray[DTYPE_t, ndim=2] img1 not None, np.ndarray[DTYPE_t, ndim=2] img2 not None, imgsize, imx):
#     filter_3(<unsigned char *>img1.data, <unsigned char *>img2.data, <int>imgsize, <int>imx)
 
 
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