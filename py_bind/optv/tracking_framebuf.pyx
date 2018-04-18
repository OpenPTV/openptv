# Implementation of the trackin_frame_buf minimal interface.

from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

from optv.vec_utils cimport vec3d, vec_copy

cdef extern from "optv/tracking_frame_buf.h":
    int c_read_targets "read_targets" (target buffer[], \
        char* file_base, int frame_num)
    int write_targets(target buffer[], int num_targets, char* file_base, \
        int frame_num)
    
    void frame_init(frame *new_frame, int num_cams, int max_targets)
    void free_frame(frame *self)
    int read_frame(frame *self, char *corres_file_base, char *linkage_file_base,
        char *prio_file_base, char **target_file_base, int frame_num)

cdef extern from "optv/correspondences.h":
    void quicksort_target_y(target *pix, int num)

DEF MAX_TARGETS = 20000 # Until improvement of read_targets to auto-allocate.
ctypedef np.float64_t pos_t

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
    """
    Represents an array of targets. Allows indexing and iteration.
    """
    def __init__(self, int size=0):
        """
        Arguments:
        size - if >0, allocates an empty target array (which should be filled
            by iteration later), otherwise nothing is allocated.
        """
        cdef target *tarr
        if size <=0:
            tarr = NULL
            size = 0
        else:
            tarr = <target *>malloc(size * sizeof(target))
        self.set(tarr, size, 1)

    # Assumed to not own the data.
    cdef void set(TargetArray self, target* tarr, int num_targets, int owns_data):
        self._tarr = tarr
        self._num_targets = num_targets
        self._owns_data = owns_data
    
    def sort_y(self):
        """
        Sorts the targets in-place by their Y coordinate. This is required for
        tracking (and relied on by OpenPTV-Python, so a useful step for those
        who need backwards-compatible output). Also renumbers the targets'
        ``pnr `` property to the new sort order.
        """
        quicksort_target_y(self._tarr, self._num_targets)
        for tnum in range(self._num_targets):
            self._tarr[tnum].pnr = tnum
        
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

cdef class Frame:
    """
    Holds a frame of particles, each with 3D position, tracking information and
    2D tracking data. 
    """
    def __init__(Frame self, num_cams, corres_file_base=None, 
        linkage_file_base=None, prio_file_base=None, target_file_base=None,
        frame_num=None):
        """
        Generate either a Frame filled by data read from text files in the
        traditional format, or a dummy object to be filled later. Dummy is
        chosen if either of corres_file_base, linkage_file_base, 
        target_file_base or frame_num is None.
        
        Arguments:
        corres_file_base, linkage_file_base - base names of the output
            correspondence and likage files respectively, to which a frame 
            number is added. Without separator.
        prio_file_base - for the linkage file with added 'prio' column.
        target_file_base - an array of strings following the same rules as
            for other *_file_base; one for each camera in the frame.
        frame_num - number of frame to add to *_file_base. A value of 0 or less
            means that no frame number should be added. The '.' separator is 
            added between the name and the frame number.
        """
        self._num_cams = num_cams
        if corres_file_base is None or linkage_file_base is None or \
            target_file_base is None or frame_num is None:
            self._frm = NULL

        else:
            self.read(corres_file_base, linkage_file_base, 
                target_file_base, frame_num, prio_file_base)
    
    def read(Frame self, char *corres_file_base, char *linkage_file_base,
        list target_file_base, int frame_num, prio_file_base):
        """
        Reads frame data from traditional text files.
        
        Arguments:
        corres_file_base, linkage_file_base - base names of the output
            correspondence and likage files respectively, to which a frame 
            number is added. Without separator.
        target_file_base - an array of strings following the same rules as
            for other *_file_base; one for each camera in the frame.
        frame_num - number of frame to add to *_file_base. A value of 0 or less
            means that no frame number should be added. The '.' separator is 
            added between the name and the frame number.
        prio_file_base - optional, for the linkage file with added 'prio'
            column.
        """
        cdef char **targ_fb = <char **> malloc(self._num_cams*sizeof(char *))
        cdef char* pb
        
        for cam in range(self._num_cams):
            targ_fb[cam] = target_file_base[cam]
        
        if prio_file_base is None:
            pb = NULL
        else:
            pb = prio_file_base
        
        if self._frm == NULL:
            self._frm = <frame *> malloc(sizeof(frame))
        else:
            # free existing substructures because allocating new ones.
            free_frame(self._frm)
        
        frame_init(self._frm, self._num_cams, MAX_TARGETS)
        success = read_frame(self._frm, corres_file_base, linkage_file_base, 
            pb, targ_fb, frame_num)
        
        return success
     
    def positions(Frame self):
        """
        Returns an (n,3) array for the 3D positions on n particles in the 
        frame.
        """
        cdef: 
            np.ndarray[ndim=2, dtype=pos_t] pos3d
            double *vec
        
        pos3d = np.empty((self._frm.num_parts, 3))
        for pt in range(self._frm.num_parts):
            vec = <double *>np.PyArray_GETPTR2(pos3d, pt, 0)
            vec_copy(vec, self._frm.path_info[pt].x)
        
        return pos3d
    
    def target_positions_for_camera(self, int cam):
        """
        Gets all targets in this frame as seen by the selected camere. The 
        targets are returned in the order corresponding to the particle order
        returned by ``positions()``.
        
        Arguments:
        int cam - camera number, starting from 0.
        
        Returns:
        an (n,2) array with the 2D position of targets detected in the image
            seen by camera ``cam``. for each 3D position. If no target in this
            camera belongs to the 3D position, its target is set to NaN. 
        """
        cdef:
            np.ndarray[ndim=2, dtype=pos_t] pos2d
            int tix
        
        pos2d = np.empty((self._frm.num_parts, 2))
        for pt in range(self._frm.num_parts):
            tix = self._frm.correspond[pt].p[cam]
            
            if tix == CORRES_NONE:
                pos2d[pt] = np.nan
            else:
                pos2d[pt,0] = self._frm.targets[cam][tix].x
                pos2d[pt,1] = self._frm.targets[cam][tix].y
        
        return pos2d
    
    def __dealloc__(self):
        if self._frm == NULL:
            return
        
        free_frame(self._frm)
        free(self._frm)
        self._frm = NULL

