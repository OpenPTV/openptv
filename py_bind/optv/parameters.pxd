#Cython definitions for parameters.h
cdef extern from "optv/parameters.h":
    ctypedef struct mm_np:
        int nlay
        double n1
        double n2[3]
        double d[3]
        double n3
        int lut
        
cdef class MultimediaParams:
    cdef mm_np* _mm_np
    