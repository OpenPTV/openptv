# Cython definitions for parameters.h
cdef extern from "optv/parameters.h":
    ctypedef struct mm_np:
        int nlay
        double n1
        double n2[3]
        double d[3]
        double n3
        int lut
    
    ctypedef struct track_par:
        double dacc, dangle, dvxmax, dvxmin
        double dvymax, dvymin, dvzmax, dvzmin
        int dsumg, dn, dnx, dny, add
    
    ctypedef struct sequence_par:
        char ** img_base_name
        int first, last
        
cdef class MultimediaParams:
    cdef mm_np* _mm_np
    
cdef class TrackingParams:
    cdef track_par * _track_par
  
cdef class SequenceParams:
    cdef sequence_par * _sequence_par


