#Cython definitions for parameters.h
cdef extern from "optv/parameters.h":
    ctypedef struct mm_np:
        int nlay
        double n1
        double n2[3]
        double d[3]
        double n3
        int lut

    ctypedef struct shaking_par:
        int seq_first
        int seq_last
        int max_shaking_points
        int max_shaking_frames
    
cdef class MultimediaParams:
    cdef mm_np* _mm_np

cdef class ShakingParams:
    cdef shaking_par* _shaking_par
    #cdef shaking_par* read_shaking_par(char * file_name)
    