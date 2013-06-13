# Cython definitions for tracking_frame_buf.h
# Implementing a minimal Python binding for frame and target.
# With time, this will grow to encompass what's needed.

cdef extern from "optv/tracking_frame_buf.h":
    ctypedef struct target:
        int pnr
        double x, y
        int n, nx, ny, sumg
        int tnr
    
cdef class Target:
    cdef target* _targ
    cdef int _owns_data
    cdef void set(Target self, target* targ)

cdef class TargetArray:
    cdef target* _tarr
    cdef int _num_targets
    cdef int _owns_data
    
    cdef void set(TargetArray self, target* tarr, int num_targets,
        int owns_data)

