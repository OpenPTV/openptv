# Implementation of the trackin_frame_buf minimal interface.

from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

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
    void fb_init(framebuf *new_buf, int buf_len, int num_cams, int max_targets,
        char *corres_file_base, char *linkage_file_base, char *prio_file_base,
        char **target_file_base)

cdef extern from "optv/correspondences.h":
    void quicksort_target_y(target *pix, int num)

DEF MAX_TARGETS = 20000 # Until improvement of read_targets to auto-allocate.
ctypedef np.float64_t pos_t

cdef class Target:
    def __cinit__(self, pnr=0, tnr=0, x=0.0, y=0.0, n=0, nx=0, ny=0, sumg=0):
        self._targ = <target*>malloc(sizeof(target))
        if self._targ != NULL:
            self._targ.pnr = pnr
            self._targ.tnr = tnr
            self._targ.x = x
            self._targ.y = y
            self._targ.n = n
            self._targ.nx = nx
            self._targ.ny = ny
            self._targ.sumg = sumg
        self._owns_data = 1
    cdef void set(self, target* targ):
        if self._owns_data == 1 and self._targ != NULL:
            free(self._targ)
        self._targ = targ
        self._owns_data = 0
    def pnr(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return self._targ.pnr
    def tnr(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return self._targ.tnr
    def x(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return self._targ.x
    def y(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return self._targ.y
    def pos(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return (self._targ.x, self._targ.y)
    def count_pixels(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return (self._targ.n, self._targ.nx, self._targ.ny)
    def sum_grey_value(self):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        return self._targ.sumg
    def set_pos(self, pos):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        self._targ.x = pos[0]
        self._targ.y = pos[1]
    def set_pnr(self, int pnr):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        self._targ.pnr = pnr
    def set_pixel_counts(self, int n, int nx, int ny):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        self._targ.n = n
        self._targ.nx = nx
        self._targ.ny = ny
    def set_sum_grey_value(self, int sumg):
        if self._targ == NULL:
            raise ValueError("Target pointer is NULL")
        self._targ.sumg = sumg
    def __dealloc__(self):
        if self._owns_data == 1 and self._targ != NULL:
            free(self._targ)
        self._targ = NULL

cdef class FrameBuf:
    cdef framebuf* _fb
    def __cinit__(self, *args):
        cdef int buf_len
        cdef int num_cams
        cdef int max_targets
        cdef char* c_corres
        cdef char* c_linkage
        cdef char* c_prio
        cdef char** c_targets
        self._fb = <framebuf*>malloc(sizeof(framebuf))
        if len(args) == 1 and isinstance(args[0], str):
            # Backward compatible: single file (not recommended for real data)
            pass  # TODO: implement if needed
        elif len(args) == 4:
            corres_file_base, linkage_file_base, target_file_base, frame_num = args
            buf_len = 1
            num_cams = len(target_file_base)
            max_targets = 20000
            py_corres = corres_file_base.encode('utf-8')
            py_linkage = linkage_file_base.encode('utf-8')
            c_corres = <char*>py_corres
            c_linkage = <char*>py_linkage
            c_prio = NULL  # Not used in this test
            py_targets = [cam.encode('utf-8') for cam in target_file_base]
            c_targets = <char**>malloc(num_cams * sizeof(char*))
            for i in range(num_cams):
                c_targets[i] = <char*>py_targets[i]
            fb_init(self._fb, buf_len, num_cams, max_targets,
                   c_corres, c_linkage, c_prio, c_targets)
        else:
            raise ValueError("FrameBuf expects either a single file or (corres_file_base, linkage_file_base, target_file_base, frame_num)")
    def num_frames(self):
        if self._fb == NULL:
            return 0
        n = fb_num_frames(self._fb)
        if n < 0:
            return 0
        return n
    def current_frame(self):
        if self._fb == NULL:
            return -1
        return fb_current_frame(self._fb)
    def next_frame(self):
        if self._fb == NULL:
            return 0
        return fb_next_frame(self._fb)
    def num_targets(self):
        if self._fb == NULL:
            return 0
        n = fb_num_targets(self._fb)
        if n < 0:
            return 0
        return n
    def get_targets(self):
        n = self.num_targets()
        if n <= 0:
            return []
        return [self._wrap_target(fb_get_target(self._fb, i)) for i in range(n)]
    cdef Target _wrap_target(self, target* t):
        if t == NULL:
            return Target()
        obj = Target()
        obj.set(t)
        return obj
    def __dealloc__(self):
        if self._fb != NULL:
            free(self._fb)
            self._fb = NULL

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
            np.ndarray[pos_t, ndim=2] pos3d
            double *vec
        
        pos3d = np.empty((self._frm.num_parts, 3), dtype=np.float64)
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
            np.ndarray[pos_t, ndim=2] pos2d
            int tix
        
        pos2d = np.empty((self._frm.num_parts, 2), dtype=np.float64)
        for pt in range(self._frm.num_parts):
            tix = self._frm.correspond[pt].p[cam]
            
            if tix == -1:  # CORRES_NONE is -1
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

