# Implementation of the trackin_frame_buf minimal interface.

from libc.stdlib cimport malloc, free

cdef extern from "optv/tracking_frame_buf.h":
    int c_read_targets "read_targets" (target buffer[], \
        char* file_base, int frame_num)

DEF MAX_TARGETS = 20000 # Until improvement of read_targets to auto-allocate.

cdef class Target:
    def __init__(self, **kwd):
        """
        Initialises the Target instance: either as an empy one (to be set later
        from C using set() ); or with initial values from Python. For the first
        case, do not supply arguments to inin. For the later case, supply all
        arguments listed below as keyword arguments.
        
        At the time of writing, the second case appears to be needed only for
        testing.
        
        Arguments:
        pnr - integer, used by the tracking code.
        x, y - doubles, target position in its 2D image, in pixels.
        n, nx, ny - number of pixels in target (total, width, height)
        tnr - used by the tracking code.
        """
        if len(kwd.keys()) == 0:
            self._owns_data = 0
            return
        
        self._owns_data = 1
        self._targ = <target *>malloc(sizeof(target))
        
        self._targ[0].pnr = kwd['pnr']
        self._targ[0].tnr = kwd['tnr']
        self._targ[0].x = kwd['x']
        self._targ[0].y = kwd['y']
        self._targ[0].n = kwd['n']
        self._targ[0].nx = kwd['nx']
        self._targ[0].ny = kwd['ny']
        self._targ[0].sumg = kwd['sumg']
    
    def __dealloc__(self):
        if self._owns_data == 1:
            free(self._targ)
    
    cdef void set(Target self, target* targ):
        self._owns_data = 0
        self._targ = targ
    
    def tnr(self):
        return self._targ[0].tnr
    
    def pos(self):
        return self._targ[0].x, self._targ[0].y

cdef class TargetArray:
    # Assumed to not own the data.
    cdef void set(TargetArray self, target* tarr, int num_targets, int owns_data):
        self._tarr = tarr
        self._num_targets = num_targets
        self._owns_data = owns_data
    
    def __getitem__(self, int ix):
        """
        Creates a Python Target instance from the C target at `ix` and returns
        it.
        
        Arguments:
        ix - integer, index into the target array.
        
        Returns:
        a Python representation of a target, using an instance of the Target
        class.
        """
        cdef Target ret
        
        if ix >= self._num_targets or ix < 0:
            raise IndexError
        
        ret = Target()
        ret.set(self._tarr + ix)
        return ret
    
    def __len__(self):
        return self._num_targets
    
    def __dealloc__(self):
        if self._owns_data == 1:
            free(self._tarr)
    
def read_targets(basename, int frame_num):
    """
    Reads a targets file and returns the targets within.
    
    Arguments:
    basename - Beginning of the image file name, to which the frame number and
        target suffix are added.
    frame_num - frame number to add to base name when creating the file name to
        open.
    
    Returns:
    A TargetArray object pointing to the read array.
    """
    cdef:
        int num_targets
        target *tarr = <target *>malloc(MAX_TARGETS * sizeof(target))
        TargetArray ret = TargetArray()
        char* c_string
    
    # Using Python strings requires some boilerplate:
    py_byte_string = basename.encode('UTF-8')
    c_string = py_byte_string
    
    num_targets = c_read_targets(tarr, c_string, frame_num)
    ret.set(tarr, num_targets, 1)
    
    return ret

