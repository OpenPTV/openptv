# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as np
from numpy cimport PyArray_SimpleNewFromData, npy_intp, NPY_DOUBLE, NPY_INT
np.import_array()  # Important! Initialize NumPy C-API

from cpython.ref cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free
from libc.string cimport strncpy

# Declare the NumPy C-API function we need
cdef extern from "numpy/arrayobject.h":
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj) except -1

cdef wrap_1d_c_arr_as_ndarray(object base_obj, int arr_size, int num_type, const void* data, int copy):
    """
    Returns as NumPy array a value internally represented as a C array.
    
    Arguments:
    object base_obj - the object supplying the memory. Needed for refcounting.
    int arr_size - length of the C array.
    int num_type - type number as required by PyArray_SimpleNewFromData
    const void* data - the underlying C array.
    int copy - 0 to return an ndarray whose data is the original data,
        1 to return a copy of the data.
    
    Returns:
    ndarray of length arr_size
    """
    cdef:
        np.npy_intp shape[1]  # for 1 dimensional array
        np.ndarray ndarr
    
    shape[0] = <np.npy_intp> arr_size
    ndarr = <np.ndarray>PyArray_SimpleNewFromData(1, shape, num_type, <void*>data)
    
    if not copy:
        Py_INCREF(base_obj)
        PyArray_SetBaseObject(ndarr, <PyObject*>base_obj)
    
    if copy:
        ndarr = ndarr.copy()
    
    return ndarr

cdef class MultimediaParams:   
    def __init__(self, n1=1.0, n2=None, n3=1.0, d=None):
        """
        Arguments (all optional):
        n1 - index of refraction of the first medium (air = 1.0)
        n2 - sequence of indices of refraction of intermediate layers
        n3 - index of refraction of the last medium
        d - sequence of thickness values for intermediate layers
        """
        self._mm_np = <mm_np *> malloc(sizeof(mm_np))
        if self._mm_np is NULL:
            raise MemoryError("Failed to allocate multimedia parameters")
        
        # Set default values
        self._mm_np[0].n1 = 1.0 if n1 is None else n1
        self._mm_np[0].n3 = 1.0 if n3 is None else n3
        self._mm_np[0].nlay = 0
        
        # Initialize arrays with zeros
        for i in range(3):
            self._mm_np[0].n2[i] = 0.0
            self._mm_np[0].d[i] = 0.0
        
        # Only process n2 and d if both are provided
        if n2 is not None and d is not None:
            if len(n2) != len(d):
                raise ValueError("Length of n2 and d must be equal")
            if len(n2) > 3:
                raise ValueError("Maximum number of layers (3) exceeded")
            
            self._mm_np[0].nlay = len(n2)
            for i in range(len(n2)):
                self._mm_np[0].n2[i] = n2[i]
                self._mm_np[0].d[i] = d[i]
             
    cdef void set_mm_np(self, mm_np * other_mm_np_c_struct):
        free(self._mm_np)
        self._mm_np = other_mm_np_c_struct
        
    def get_nlay(self):
        return self._mm_np[0].nlay
        
    def get_n1(self):
        return self._mm_np[0].n1
    
    def set_n1(self, n1):
        self._mm_np[0].n1 = n1
        
    def set_layers(self, refr_index, thickness):
        if len(refr_index) != len(thickness):
            raise ValueError("Lengths of refractive index and thickness must be equal.")
        else:
            for i in range(len(refr_index)):
                self._mm_np[0].n2[i] = refr_index[i]
            
            for i in range(len(thickness)):
                self._mm_np[0].d[i] = thickness[i]
            
            self._mm_np[0].nlay = len(refr_index)

    def get_n2(self, copy=True):
        """
        Arguments:
        copy - False for returned numpy object to take ownership of 
            the memory of the n2 field of the underlying mm_np struct. True
            (default) to return a COPY instead.
        """
        cdef int arr_size = <int>(sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0]))
        return wrap_1d_c_arr_as_ndarray(self, arr_size, NPY_DOUBLE, 
            self._mm_np[0].n2, (1 if copy else 0))
            
    def get_d(self, copy=True):
        """
        Arguments:
        copy - False for returned numpy object to take ownership of 
            the memory of the d field of the underlying mm_np struct. True
            (default) to return a COPY instead.
        """
        cdef int arr_size = <int>(sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0]))
        return wrap_1d_c_arr_as_ndarray(self, arr_size, NPY_DOUBLE, 
            self._mm_np[0].d, (1 if copy else 0))
            
    def get_n3(self):
        return self._mm_np[0].n3
        
    def set_n3(self, n3):
        self._mm_np[0].n3 = n3

    def __richcmp__(MultimediaParams self, MultimediaParams other, int operator):
        cdef int c_compare_result = compare_mm_np(self._mm_np, other._mm_np)
        if operator == 2:  # "==" action was performed
            return c_compare_result != 0
        elif operator == 3:  # "!=" action was performed
            return c_compare_result == 0
        else:
            raise TypeError("Unhandled comparison operator")

    def __dealloc__(self):
        free(self._mm_np)

cdef class TrackingParams:   
    """
    Wrapping the track_par C struct for pythonic access
    Binding the read_track_par C function
    Objects of this type can be checked for equality using "==" and "!=" operators
    """
    def __init__(self, **kwds):
        """
        Arguments (all optional):
        accel_lim - the limit on the norm of acceleration, [L/frame**2]
        angle_lim - the limit on angle between two path links, [gon]
        velocity_lims - a 3x2 array, the min and max velocity in each axis,
            [L/frame]
        add_particle - whether or not to add a new particle in candidate
            positions that have enough targets but no matching correspondence
            result (boolean int).
        """
        self._track_par = < track_par *> malloc(sizeof(track_par))
        
        if 'accel_lim' in kwds:
            self.set_dacc(kwds['accel_lim'])
        if 'angle_lim' in kwds:
            self.set_dangle(kwds['angle_lim'])
        if 'velocity_lims' in kwds:
            self.set_dvxmax(kwds['velocity_lims'][0][1])
            self.set_dvxmin(kwds['velocity_lims'][0][0])
            self.set_dvymax(kwds['velocity_lims'][1][1])
            self.set_dvymin(kwds['velocity_lims'][1][0])
            self.set_dvzmax(kwds['velocity_lims'][2][1])
            self.set_dvzmin(kwds['velocity_lims'][2][0])
        if 'add_particle' in kwds:
            self.set_add(kwds['add_particle'])
            
        # The rest of the members are not used in the current algorithm.
        self.set_dsumg(0)
        self.set_dn(0)
        self.set_dnx(0)
        self.set_dny(0)
        
    # Reads tracking parameters from a config file with the 
    # following format: each line is a value, in this order:
    # 1. dvxmin
    # 2. dvxmax
    # 3. dvymin
    # 4. dvymax
    # 5. dvzmin
    # 6. dvzmax
    # 7. dangle
    # 8. dacc
    # 9. add
    #
    # Argument: 
    # file_name - path to the text file containing the parameters.
      
    def read_track_par(self, str filename):
        """
        Reads tracking parameters from an old-style .par file having the
        objects' arguments ordered one per line.
        """
        self._filename_bytes = filename.encode('utf-8')
        # Cast away const
        cdef char* c_filename = <char*><char*>self._filename_bytes
        free(self._track_par)
        self._track_par = read_track_par(c_filename)
        
        # Ensure the file is properly closed after reading
        if hasattr(self, '_file') and self._file is not None:
            self._file.close()
    
    # Checks for equality between this and other trackParams objects
    # Gives the ability to use "==" and "!=" operators on two trackParams objects
    def __richcmp__(TrackingParams self, TrackingParams other, int operator):
        cdef int c_compare_result = compare_track_par(self._track_par, other._track_par)
        if operator == 2:  # "==" action was performed
            return c_compare_result != 0
        elif operator == 3:  # "!=" action was performed
            return c_compare_result == 0
        else:
            raise TypeError("Unhandled comparison operator")

    # Getters and setters    
    def get_dacc(self):
        return self._track_par[0].dacc
    
    def set_dacc(self, dacc):
        self._track_par[0].dacc = dacc

    def get_dangle(self):
        """
        The angle limit in [gon] (gradians)
        """
        return self._track_par[0].dangle
    
    def set_dangle(self, dangle):
        """
        Set the angle limit for tracking, in [gon] (gradians)
        """
        self._track_par[0].dangle = dangle
        
    def get_dvxmax(self):
        return self._track_par[0].dvxmax
    
    def set_dvxmax(self, dvxmax):
        self._track_par[0].dvxmax = dvxmax
    
    def get_dvxmin(self):
        return self._track_par[0].dvxmin
    
    def set_dvxmin(self, dvxmin):
        self._track_par[0].dvxmin = dvxmin
        
    def get_dvymax(self):
        return self._track_par[0].dvymax
    
    def set_dvymax(self, dvymax):
        self._track_par[0].dvymax = dvymax
        
    def get_dvymin(self):
        return self._track_par[0].dvymin
    
    def set_dvymin(self, dvymin):
        self._track_par[0].dvymin = dvymin
        
    def get_dvzmax(self):
        return self._track_par[0].dvzmax
    
    def set_dvzmax(self, dvzmax):
        self._track_par[0].dvzmax = dvzmax
        
    def get_dvzmin(self):
        return self._track_par[0].dvzmin
    
    def set_dvzmin(self, dvzmin):
        self._track_par[0].dvzmin = dvzmin
        
    def get_dsumg(self):
        return self._track_par[0].dsumg
    
    def set_dsumg(self, dsumg):
        self._track_par[0].dsumg = dsumg
        
    def get_dn(self):
        return self._track_par[0].dn
    
    def set_dn(self, dn):
        self._track_par[0].dn = dn
        
    def get_dnx(self):
        return self._track_par[0].dnx
    
    def set_dnx(self, dnx):
        self._track_par[0].dnx = dnx
        
    def get_dny(self):
        return self._track_par[0].dny
    
    def set_dny(self, dny):
        self._track_par[0].dny = dny
        
    def get_add(self):
        return self._track_par[0].add
    
    def set_add(self, add):
        self._track_par[0].add = add
        
    # Memory freeing
    def __dealloc__(self):
        free(self._track_par)
        
# Wrapping the sequence_par C struct (declared in liboptv/paramethers.h) for pythonic access
# Binding the read_track_par C function
# Objects of this type can be checked for equality using "==" and "!=" operators

cdef class SequenceParams:
    """
    Wrapping the sequence_par C struct (declared in liboptv/paramethers.h) for 
    pythonic access. Binding the read_track_par C function. Objects of this 
    type can be checked for equality using "==" and "!=" operators.
    """
    def __init__(self, **kwargs):
        """
        Arguments (all optional, but either num_cams or image_base required):
        num_cams - number of cameras used in the scene.
        image_base - a list of image base names, to which the frame number 
            is added during sequence operations.
        frame_range - (first, last)
        """
        cdef int num_cams
        
        if 'num_cams' in kwargs:
            num_cams = kwargs['num_cams']
        elif 'image_base' in kwargs:
            num_cams = len(kwargs['image_base'])
        else:
            raise ValueError(
                "SequenceParams requires either num_cams or image_base")
        
        self._sequence_par = new_sequence_par(num_cams)
        if self._sequence_par is NULL:
            raise MemoryError("Failed to allocate sequence parameters")
        
        if 'frame_range' in kwargs:
            self.set_first(kwargs['frame_range'][0])
            self.set_last(kwargs['frame_range'][1])
        if 'image_base' in kwargs:
            for cam in range(num_cams):
                self.set_img_base_name(cam, kwargs['image_base'][cam])
    
    def get_first(self):
        return self._sequence_par[0].first
    
    def set_first(self, first):
        self._sequence_par[0].first = first
        
    def get_last(self):
        return self._sequence_par[0].last
    
    def set_last(self, last):
        self._sequence_par[0].last = last
        
    def read_sequence_par(self, str filename, int num_cams):
        """Read sequence parameters from a text file.
        
        Arguments:
            filename: path to the parameter file
            num_cams: number of cameras
        """
        self._filename_bytes = filename.encode('utf-8')
        cdef char* c_filename = <char*><char*>self._filename_bytes
        
        if self._sequence_par != NULL:
            free_sequence_par(self._sequence_par)
        
        self._sequence_par = read_sequence_par(c_filename, num_cams)
        
    # Get image base name of camera #cam
    def get_img_base_name(self, cam):
        """Get image base name of camera #cam"""
        cdef char * c_str = self._sequence_par[0].img_base_name[cam]
        if c_str is NULL:
            return None
        return c_str.decode('utf-8')
    
    # Set image base name for camera #cam
    def set_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char * c_string = py_byte_string
        strncpy(self._sequence_par[0].img_base_name[cam], c_string, len(new_img_name) + 1)
    
    # Checks for equality between this and other SequenceParams objects
    # Gives the ability to use "==" and "!=" operators on two SequenceParams objects
    def __richcmp__(SequenceParams self, SequenceParams other, int operator):
        cdef int c_compare_result = compare_sequence_par(self._sequence_par, other._sequence_par)
        if operator == 2:  # "==" action was performed
            return c_compare_result != 0
        elif operator == 3:  # "!=" action was performed
            return c_compare_result == 0
        else:
            raise TypeError("Unhandled comparison operator")

    def __dealloc__(self):
        """Free sequence_par struct."""
        if self._sequence_par is not NULL:
            free_sequence_par(self._sequence_par)
            self._sequence_par = NULL
        
cdef class VolumeParams:
    """
    Wrapping the volume_par C struct (declared in liboptv/paramethers.h) for 
    pythonic access. Objects of this type can be checked for equality using 
    "==" and "!=" operators.
    """
    def __init__(self, **kwargs):
        """
        Accepted keyword arguments:
        x_span - min. and max. value of X in the search volume, (2,) array.
        z_spans - list of 2 (2,) arrays, each with min. and max. value of Z 
            in the search volume.
        pixels_tot, pixels_x, pixels_y - min. ratio of pixel counts between a 
            target and a candidate: total and per image dimension.
        ref_gray - min. ratio of sum of grey values for a target and candidate.
        epipolar_band - width of epipolar line, or the distance from the 
            epipolar line where targets are taken as candidates.
        min_correlation - minimum value for the correlation between particles,
            calculated from match of the different target measures to the 
            reference values in this object.
        """
        self._volume_par = < volume_par *> malloc(sizeof(volume_par))
        
        if 'x_span' in kwargs:
            self.set_X_lay(kwargs['x_span'])
        if 'z_spans' in kwargs:
            mins, maxs = zip(*kwargs['z_spans']) # Python's transpose :)
            self.set_Zmin_lay(mins)
            self.set_Zmax_lay(maxs)
        
        if 'pixels_x' in kwargs:
            self.set_cnx(kwargs['pixels_x'])
        if 'pixels_y' in kwargs:
            self.set_cny(kwargs['pixels_y'])
        if 'pixels_tot' in kwargs:
            self.set_cn(kwargs['pixels_tot'])
        
        if 'ref_gray' in kwargs:
            self.set_csumg(kwargs['ref_gray'])
        if 'min_correlation' in kwargs:
            self.set_corrmin(kwargs['min_correlation'])
        if 'epipolar_band' in kwargs:
            self.set_eps0(kwargs['epipolar_band'])
    
    # Getters and setters
    def get_X_lay(self, copy=True):
        cdef int arr_size = <int>(sizeof(self._volume_par[0].X_lay) / sizeof(self._volume_par[0].X_lay[0]))
        return wrap_1d_c_arr_as_ndarray(self, arr_size, NPY_DOUBLE, 
            self._volume_par[0].X_lay, copy)
    
    def get_Zmin_lay(self, copy=True):
        cdef int arr_size = <int>(sizeof(self._volume_par[0].Zmin_lay) / sizeof(self._volume_par[0].Zmin_lay[0]))
        return wrap_1d_c_arr_as_ndarray(self, arr_size, NPY_DOUBLE, 
            self._volume_par[0].Zmin_lay, copy)
    
    def get_Zmax_lay(self, copy=True):
        cdef int arr_size = <int>(sizeof(self._volume_par[0].Zmax_lay) / sizeof(self._volume_par[0].Zmax_lay[0]))
        return wrap_1d_c_arr_as_ndarray(self, arr_size, NPY_DOUBLE, 
            self._volume_par[0].Zmax_lay, copy)
    
    def set_X_lay(self, X_lay):
        for i in range(len(X_lay)):
            self._volume_par[0].X_lay[i] = X_lay[i]
            
    def set_Zmin_lay(self, Zmin_lay):
        for i in range(len(Zmin_lay)):
            self._volume_par[0].Zmin_lay[i] = Zmin_lay[i]
            
    def set_Zmax_lay(self, Zmax_lay):
        for i in range(len(Zmax_lay)):
            self._volume_par[0].Zmax_lay[i] = Zmax_lay[i]
            
    def get_cn(self):
        return self._volume_par[0].cn
    
    def set_cn(self, cn):
        self._volume_par[0].cn = cn
    
    def get_cnx(self):
        return self._volume_par[0].cnx
    
    def set_cnx(self, cnx):
        self._volume_par[0].cnx = cnx
        
    def get_cny(self):
        return self._volume_par[0].cny
    
    def set_cny(self, cny):
        self._volume_par[0].cny = cny
        
    def get_csumg(self):
        return self._volume_par[0].csumg
    
    def set_csumg(self, csumg):
        self._volume_par[0].csumg = csumg
        
    def get_eps0(self):
        return self._volume_par[0].eps0
    
    def set_eps0(self, eps0):
        self._volume_par[0].eps0 = eps0
        
    def get_corrmin(self):
        return self._volume_par[0].corrmin
    
    def set_corrmin(self, corrmin):
        self._volume_par[0].corrmin = corrmin
        
    def read_volume_par(self, str filename):
        """Read volume parameters from a text file.
        
        Arguments:
            filename: path to the parameter file
        """
        # Store the bytes object as an instance attribute to prevent garbage collection
        self._filename_bytes = filename.encode('utf-8')
        cdef char* c_filename = <char*><char*>self._filename_bytes
        
        # Free existing volume_par if it exists
        if self._volume_par != NULL:
            free(self._volume_par)
            
        # Read new parameters
        self._volume_par = read_volume_par(c_filename)
        if self._volume_par == NULL:
            raise IOError("Failed to read volume parameters from " + filename)
    
    # Checks for equality between self and other VolumeParams objects
    # Gives the ability to use "==" and "!=" operators on two VolumeParams objects
    def __richcmp__(VolumeParams self, VolumeParams other, int operator):
        cdef int c_compare_result = compare_volume_par(self._volume_par, other._volume_par)
        if operator == 2:  # "==" action was performed
            return c_compare_result != 0
        elif operator == 3:  # "!=" action was performed
            return c_compare_result == 0
        else:
            raise TypeError("Unhandled comparison operator")

    def __dealloc__(self):
        free(self._volume_par)
        
cdef class ControlParams:
    """
    Wrapping the control_par C struct (declared in liboptv/paramethers.h) for 
    pythonic access. Objects of this type can be checked for equality using 
    "==" and "!=" operators.
    """
    def __init__(self, int num_cams, flags=None, image_size=None, pixel_size=None,
        cam_side_n=None, wall_ns=None, wall_thicks=None, object_side_n=None):
        """
        Arguments (all optional except num_cams):
        cams - number of cameras used in the scene.
        flags - a list containing name of set flags, select from 'hp', 
            'allcam', 'headers'.
        image_size - sequence, w,h image size in pixels.
        pixel_size - sequence, w,h pixel size in mm.
        cam_side_n, wall_ns, wall_thicks, object_side_n - see MultimediaParams
        """
        if flags is None:
            flags = []
            
        # Initialize the control parameter structure
        self._control_par = new_control_par(num_cams)
        if self._control_par is NULL:
            raise MemoryError("Failed to allocate control parameters")
            
        # Set the flags
        self.set_hp_flag('hp' in flags)
        self.set_allCam_flag('allcam' in flags)
        self.set_tiff_flag('headers' in flags)
        self.set_chfield(0)  # legacy stuff
        
        # Set optional parameters
        if image_size is not None:
            self.set_image_size(image_size)
        if pixel_size is not None:
            self.set_pixel_size(pixel_size)
        
        # Initialize multimedia parameters
        self._multimedia_params = MultimediaParams(
            n1=cam_side_n, n2=wall_ns, d=wall_thicks, n3=object_side_n)
        
        # Update multimedia parameters pointer
        if self._control_par != NULL and self._control_par.mm != NULL:
            free(self._control_par.mm)
        self._control_par.mm = self._multimedia_params._mm_np
        
    # Getters and setters
    def get_num_cams(self):
        return self._control_par[0].num_cams
        
    def get_hp_flag(self):
        return self._control_par[0].hp_flag != 0
    
    def set_hp_flag(self, hp_flag):
        if hp_flag == True:
            self._control_par[0].hp_flag = 1
        else:
            self._control_par[0].hp_flag = 0
        
    def get_allCam_flag(self):
        return self._control_par[0].allCam_flag != 0
    
    def set_allCam_flag(self, allCam_flag):
        if allCam_flag == True:
            self._control_par[0].allCam_flag = 1
        else:
            self._control_par[0].allCam_flag = 0
        
    def get_tiff_flag(self):
        return self._control_par[0].tiff_flag != 0
    
    def set_tiff_flag(self, tiff_flag):
        if tiff_flag == True:
            self._control_par[0].tiff_flag = 1
        else:
            self._control_par[0].tiff_flag = 0            
    
    def get_image_size(self, copy=True):
        return (self._control_par[0].imx, self._control_par[0].imy)
    
    def set_image_size(self, image_dims_tuple):
        if len(image_dims_tuple) != 2:
            raise ValueError("Tuple passed as image size must have exactly two elements (width,height).")
        # set the values
        self._control_par[0].imx = image_dims_tuple[0]
        self._control_par[0].imy = image_dims_tuple[1]
    
    def get_pixel_size(self, copy=True):
        return (self._control_par[0].pix_x, self._control_par[0].pix_y)
    
    def set_pixel_size(self, pixel_size_tuple):
        if len(pixel_size_tuple) != 2:
            raise ValueError("Tuple passed as pixel size must have exactly two elements (width,height).")
        # set the values
        self._control_par[0].pix_x = pixel_size_tuple[0]
        self._control_par[0].pix_y = pixel_size_tuple[1]
    
    def get_chfield(self):
        return self._control_par[0].chfield
    
    def set_chfield(self, chfield):
        self._control_par[0].chfield = chfield
        
    # calls read_control_par() C function that reads general control parameters that are not present in
    # other config files but are needed generally. The arguments are read in
    # this order:
    #   
    # 1. num_cams - number of cameras in a frame.
    # 2n (n = 1..num_cams). img_base_name
    # 2n+1. cal_img_base_name
    # 2n+2. hp_flag - high pass filter flag (0/1)
    # 2n+3. allCam_flag - flag using the particles that are matched in all cameras
    # +4. tiff_flag, use TIFF headers or not (if RAW images) 0/1
    # +5. imx - horizontal size of the image/sensor in pixels, e.g. 1280
    # +6. imy - vertical size in pixels, e.g. 1024
    # +7. pix_x
    # +8. pix_y - pixel size of the sensor (one value per experiment means 
    # that all cameras are identical. TODO: allow for different cameras), in [mm], 
    # e.g. 0.010 = 10 micron pixel
    # +9. chfield - 
    # +10. mmp.n1 - index of refraction of the first media (air = 1)
    # +11. mmp.n2[0] - index of refraction of the second media - glass windows, can
    # be different?
    # +12. mmp.n3 - index of refraction of the flowing media (air, liquid)
    # 2n+13. mmp.d[0] - thickness of the glass/perspex windows (second media), can be
    # different ?
    #   
    # (if n = 4, then 21 lines)
    #   
    # Arguments:
    # filename - path to text file containing the parameters.
    def read_control_par(self, str filename):
        """Read control parameters from a text file.
        
        Arguments:
            filename: path to the parameter file
        """
        self._filename_bytes = filename.encode('utf-8')
        cdef char* c_filename = <char*><char*>self._filename_bytes
        
        if self._control_par != NULL:
            self._control_par[0].mm = NULL  # Prevent double free
            free_control_par(self._control_par)
        
        self._multimedia_params._mm_np = NULL
        self._control_par = read_control_par(c_filename)
        self._multimedia_params.set_mm_np(self._control_par[0].mm)
        
    # Get image base name of camera #cam
    def get_img_base_name(self, cam):
        """Get image base name of camera #cam"""
        cdef char * c_str = self._control_par[0].img_base_name[cam]
        if c_str is NULL:
            return None
        return c_str.decode('utf-8')
    
    # Set image base name for camera #cam
    def set_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char * c_string = py_byte_string
        strncpy(self._control_par[0].img_base_name[cam], c_string, len(new_img_name) + 1)
    
    # Get calibration image base name of camera #cam
    def get_cal_img_base_name(self, cam):
        """Get calibration image base name of camera #cam"""
        cdef char * c_str = self._control_par[0].cal_img_base_name[cam]
        if c_str is NULL:
            return None
        return c_str.decode('utf-8')
    
    # Set calibration image base name for camera #cam
    def set_cal_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char * c_string = py_byte_string
        strncpy(self._control_par[0].cal_img_base_name[cam], c_string, len(new_img_name) + 1)
    
    def get_multimedia_params(self):
        return self._multimedia_params
    
    # Checks for equality between this and other ControlParams objects
    # Gives the ability to use "==" and "!=" operators on two ControlParams objects
    def __richcmp__(ControlParams self, ControlParams other, int operator):
        cdef int c_compare_result = compare_control_par(self._control_par, other._control_par)
        if operator == 2:  # "==" action was performed
            return c_compare_result != 0
        elif operator == 3:  # "!=" action was performed
            return c_compare_result == 0
        else:
            raise TypeError("Unhandled comparison operator")

    def __dealloc__(self):
        """Deallocate control_par struct."""
        if self._control_par != NULL:
            # set the mm pointer to NULL to prevent c_free_control_par 
            # from freeing it. MultimediaParams object will free this
            # memory in its python destructor when there will be no references to it.
            self._control_par[0].mm = NULL
            free_control_par(self._control_par)  # Use the declared free function
            self._control_par = NULL

cdef class TargetParams:
    """
    Wrapping the target_par C struct (declared in liboptv/paramethers.h) for 
    pythonic access. Objects of this type can be checked for equality using 
    "==" and "!=" operators.
    """
    def __init__(self, int discont=0, gvthresh=None, 
        pixel_count_bounds=(0, 1000),
        xsize_bounds=(0, 100), ysize_bounds=(0, 100), int min_sum_grey=0, 
        int cross_size=2):
        """
        Arguments (all optional):
        int discont - maximum discontinuity parameter.
        gvthresh - sequence, per-camera grey-level threshold beneath which a 
            pixel is not considered. Currently limited to 4 cameras.
        pixel_count_bounds - tuple, min and max number of pixels in a target.
        xsize_bounds, ysize_bounds - each a tuple, min and max size of target
            in the respective dimension.
        int min_sum_grey - minimal sum of grey values in a target.
        int cross_size - legacy parameter, don't use.
        """
        if gvthresh is None:
            gvthresh = [0] * 4
        
        self._targ_par = <target_par *> malloc(sizeof(target_par))
        
        self.set_max_discontinuity(discont)
        self.set_grey_thresholds(gvthresh)
        self.set_pixel_count_bounds(pixel_count_bounds)
        self.set_xsize_bounds(xsize_bounds)
        self.set_ysize_bounds(ysize_bounds)
        self.set_min_sum_grey(min_sum_grey)
        self.set_cross_size(cross_size)
    
    def get_max_discontinuity(self):
        return self._targ_par.discont
    
    def set_max_discontinuity(self, int discont):
        self._targ_par.discont = discont
    
    def get_grey_thresholds(self, num_cams=4, copy=True):
        """
        Get the first ``num_cams`` (up to 4) per-camera grey-level thresholds.
        
        Arguments:
        num_cams - the number of cameras to get. Currently cannot exceed 4
            (raises ValueError). Default is 4.
        copy - if True, return a copy of the underlying array. This way the 
            original is safe.
        """
        return wrap_1d_c_arr_as_ndarray(self, num_cams, NPY_INT, 
            self._targ_par.gvthres, (1 if copy else 0))
        
    def set_grey_thresholds(self, gvthresh):
        """
        Arguments:
        gvthresh - a sequence of at most 4 ints.
        """
        if len(gvthresh) > 4:
            raise ValueError("Can't store more than 4 grey-level thresholds.")
        
        for gvx in xrange(len(gvthresh)):
            self._targ_par.gvthres[gvx] = gvthresh[gvx]
    
    def get_pixel_count_bounds(self):
        """
        Returns a tuple (min, max) pixels per target.
        """
        return (self._targ_par.nnmin, self._targ_par.nnmax)
    
    def set_pixel_count_bounds(self, bounds):
        """
        Arguments:
        bounds - tuple, min and max number of pixels in a target.
        """
        self._targ_par.nnmin = bounds[0]
        self._targ_par.nnmax = bounds[1]
    
    def get_xsize_bounds(self):
        """
        Returns a tuple (min, max) x size of a target.
        """
        return (self._targ_par.nxmin, self._targ_par.nxmax)
    
    def set_xsize_bounds(self, bounds):
        """
        Arguments:
        bounds - tuple, min and max x size of a target.
        """
        self._targ_par.nxmin = bounds[0]
        self._targ_par.nxmax = bounds[1]
        
    def get_ysize_bounds(self):
        """
        Returns a tuple (min, max) y size of a target.
        """
        return (self._targ_par.nymin, self._targ_par.nymax)
    
    def set_ysize_bounds(self, bounds):
        """
        Arguments:
        bounds - tuple, min and max x size of a target.
        """
        self._targ_par.nymin = bounds[0]
        self._targ_par.nymax = bounds[1]
    
    def get_min_sum_grey(self):
        return self._targ_par.sumg_min
    
    def set_min_sum_grey(self, int min_sumg):
        self._targ_par.sumg_min = min_sumg
    
    def get_cross_size(self):
        return self._targ_par.cr_sz
    
    def set_cross_size(self, int cr_sz):
        self._targ_par.cr_sz = cr_sz
    
    def read(self, str inp_filename):
        """
        Reads target recognition parameters from a legacy .par file, which 
        holds one parameter per line. The arguments are read in this order:
        
        1. gvthres[0]
        2. gvthres[1]
        3. gvthres[2]
        4. gvthres[3]
        5. discont
        6. nnmin
        7. nnmax
        8. nxmin
        9. nxmax
        10. nymin
        11. nymax
        12. sumg_min
        13. cr_sz
        
        Fills up the fields of the object from the file and returns.
        """
        # Convert the string to bytes and store it
        self._filename_bytes = inp_filename.encode('utf-8')
        
        # Get a pointer to the underlying buffer
        cdef char* c_filename = <char*><char*>self._filename_bytes
        
        free(self._targ_par)
        self._targ_par = read_target_par(c_filename)
        
        if self._targ_par == NULL:
            raise IOError("Problem reading target recognition parameters.")
    
    def __dealloc__(self):
        free(self._targ_par)
