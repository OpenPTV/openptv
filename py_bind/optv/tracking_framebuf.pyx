# Implementation of the trackin_frame_buf minimal interface.

from libc.stdlib cimport malloc, free

cdef extern from "optv/tracking_frame_buf.h":
    int c_read_targets "read_targets" (target buffer[], \
        char* file_base, int frame_num)
    int write_targets(target buffer[], int num_targets, char* file_base, \
        int frame_num)

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
        
        self.set_pnr(kwd['pnr'])
        self.set_tnr(kwd['tnr'])
        self.set_pos([kwd['x'], kwd['y']])
        self.set_pixel_counts(kwd['n'], kwd['nx'], kwd['ny'])
        self.set_sum_grey_value(kwd['sumg'])
    
    def __dealloc__(self):
        if self._owns_data == 1:
            free(self._targ)
    
    cdef void set(Target self, target* targ):
        self._owns_data = 0
        self._targ = targ
    
    def tnr(self):
        return self._targ[0].tnr
    
    def set_tnr(self, tnr):
        self._targ[0].tnr = tnr
    
    def pnr(self):
        return self._targ[0].pnr
    
    def set_pnr(self, pnr):
        self._targ[0].pnr = pnr
    
    def pos(self):
        """
        Get target position - a tuple (x,y)
        """
        return self._targ[0].x, self._targ[0].y
    
    def set_pos(self, pos):
        """
        Set target position in pixel coordinates.
        
        Arguments:
        pos - a 2-element sequence, for the x and y pixel position.
        """
        self._targ[0].x = pos[0]
        self._targ[0].y = pos[1]

    def count_pixels(self):
        """
        Get the pixel counts associated with this target.
        
        Returns:
        n, nx, ny - number of pixels in target (total, width, height)
        """
        return self._targ.n, self._targ.nx, self._targ.ny

    def set_pixel_counts(self, n, nx, ny):
        """
        Set the pixel counts associated with this target.
        
        Arguments:
        n, nx, ny - number of pixels in target (total, width, height)
        """
        self._targ.n = n
        self._targ.nx = nx
        self._targ.ny = ny
    
    def sum_grey_value(self):
        """
        Returns the sum of grey values of pixels belonging to target.
        """
        return self._targ.sumg
    
    def set_sum_grey_value(self, sumg):
        """
        Returns the sum of grey values of pixels belonging to target.
        """
        self._targ.sumg = sumg

cdef class TargetArray:
    # Assumed to not own the data.
    cdef void set(TargetArray self, target* tarr, int num_targets, int owns_data):
        self._tarr = tarr
        self._num_targets = num_targets
        self._owns_data = owns_data
    
    def write(self, char *file_base, int frame_num):
        """
        Writes a _targets file - a text format for targets. First line: number
        of targets. Each following line: pnr, x, y, n, nx, ny, sumg, tnr.
        the output file name is of the form <base_name><frame>_targets.
        
        Arguments:
        file_base - path to the file, base part.
        frame_num - frame number part of the file name.
        """
        write_targets(self._tarr, self._num_targets, file_base, frame_num)

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

