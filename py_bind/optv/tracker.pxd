
from optv.parameters cimport sequence_par, track_par, volume_par, control_par
from optv.tracking_framebuf cimport framebuf, framebuf_base
from optv.calibration cimport calibration

cdef extern from "optv/tracking_run.h":
    ctypedef struct tracking_run:
        sequence_par *seq_par
        calibration **cal
        framebuf *fb
    
    tracking_run *tr_new(
        sequence_par *seq_par, track_par *tpar, volume_par *vpar, control_par *cpar, 
        calibration **cal, framebuf_base *fb, double flatten_tol);

cdef extern from "optv/track.h":
    cdef enum:
        TR_BUFSPACE, MAX_TARGETS
    void track_forward_start(tracking_run *tr)
    void trackcorr_c_loop(tracking_run *run_info, int step)
    void trackcorr_c_finish(tracking_run *run_info, int step)
    double trackback_c(tracking_run *run_info)

cdef class Tracker:
    cdef tracking_run *run_info
    cdef framebuf *fb
    cdef int step
    cdef object _keepalive

    
