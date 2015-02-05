#Cython definitions for parameters.h
cdef extern from "optv/parameters.h":
    ctypedef struct mm_np:
        int nlay
        double n1
        double n2[3]
        double d[3]
        double n3
        int lut
        
cdef class Py_mm_np:
    cdef mm_np* _mm_np
    cdef int _owns_data
    cdef void set(Py_mm_np self, mm_np* m)
