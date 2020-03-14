# -*- coding: utf-8 -*-
"""
Implementation of bindings for the tracking code.

Created on Sun Apr 23 15:41:32 2017

@author: yosef
"""

from libc.stdlib cimport free
from optv.parameters cimport ControlParams, TrackingParams, SequenceParams, \
    VolumeParams
from optv.orientation cimport cal_list2arr
from optv.tracking_framebuf cimport fb_free

default_naming = {
    'corres': b'res/rt_is',
    'linkage': b'res/ptv_is',
    'prio': b'res/added'
}

cdef class Tracker:
    """
    Workflow: instantiate, call restart() to initialize the frame buffer, then
    call either ``step_forward()`` while it still return True, then call
    ``finalize()`` to finish the run. Alternatively, ``full_forward()`` will 
    do all this for you.
    """
    def __init__(self, ControlParams cpar, VolumeParams vpar, 
        TrackingParams tpar, SequenceParams spar, list cals,
        dict naming=default_naming, flatten_tol=0.0001):
        """
        Arguments:
        ControlParams cpar, VolumeParams vpar, TrackingParams tpar, 
        SequenceParams spar - the usual parameter objects, as read from 
            anywhere.
        cals - a list of Calibratiopn objects.
        dict naming - a dictionary with naming rules for the frame buffer 
            files. See the ``default_naming`` member (which is the default).
        """
        # We need to keep a reference to the Python objects so that their
        # allocations are not freed.
        self._keepalive = (cpar, vpar, tpar, spar, cals)
        
        self.run_info = tr_new(spar._sequence_par, tpar._track_par,
            vpar._volume_par, cpar._control_par, TR_BUFSPACE, MAX_TARGETS,
            naming['corres'], naming['linkage'], naming['prio'], 
            cal_list2arr(cals), flatten_tol)
    
    def restart(self):
        """
        Prepare a tracking run. Sets up initial buffers and performs the
        one-time calculations used throughout the loop.
        """
        self.step = self.run_info.seq_par.first
        track_forward_start(self.run_info)
    
    def step_forward(self):
        """
        Perform one tracking step for the current frame of iteration.
        """
        if self.step >= self.run_info.seq_par.last:
            return False
        
        trackcorr_c_loop(self.run_info, self.step)
        self.step += 1
        return True
    
    def finalize(self):
        """
        Finish a tracking run.
        """
        trackcorr_c_finish(self.run_info, self.step)
    
    def full_forward(self):
        """
        Do a full tracking run from restart to finalize.
        """
        track_forward_start(self.run_info)
        for step in range(
                self.run_info.seq_par.first, self.run_info.seq_par.last):
            trackcorr_c_loop(self.run_info, step)
        trackcorr_c_finish(self.run_info, self.run_info.seq_par.last)
    
    def full_backward(self):
        """
        Does a full backward run on existing tracking results. so make sure
        results exist or it will explode in your face.
        """
        trackback_c(self.run_info)
        
    def current_step(self):
        return self.step
    
    def __dealloc__(self):
        # Don't call tr_free, just free the memory that belongs to us.
        fb_free(self.run_info.fb)
        free(self.run_info.cal) # allocated by cal_list2arr, leafs belong to
                                # owner of the Tracker.
        free(self.run_info) # not using tr_free() which assumes ownership of 
                            # parameter structs.
        