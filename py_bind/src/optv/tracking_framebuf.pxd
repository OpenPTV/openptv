# cython: language_level=3
# distutils: language = c

# Cython definitions for tracking_frame_buf.h
# Implementing a minimal Python binding for frame and target.

from optv.vec_utils cimport vec3d

cdef extern from "optv/tracking_frame_buf.h":
    ctypedef struct target:
        int pnr
        double x, y
        int n, nx, ny, sumg
        int tnr
    
    ctypedef struct corres:
        int nr
        int p[4]
    
    cpdef enum:
        CORRES_NONE = -1
        PT_UNUSED = -999
    
    ctypedef struct path_inf "P":
        vec3d x
        int prev, next, prio
    
    ctypedef struct frame:
        path_inf *path_info
        corres *correspond
        target **targets
        int num_cams, max_targets, num_parts
        int *num_targets
    
    ctypedef struct framebuf:
        pass
    
    void fb_free(framebuf *self)
    
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

cdef class Frame:
    cdef frame *_frm
    cdef int _num_cams # only used for dummy frames.
