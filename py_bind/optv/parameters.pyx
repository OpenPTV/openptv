# Implementation of Python binding to parameters.h
from libc.stdlib cimport malloc, free
from libc.string cimport strncpy

import numpy
numpy.import_array()

cimport numpy as numpy
from cpython cimport PyObject, Py_INCREF

cdef extern from "optv/parameters.h":
    int c_compare_mm_np "compare_mm_np"(mm_np * mm_np1, mm_np * mm_np2)
    
    track_par * c_read_track_par "read_track_par"(char * file_name)
    int c_compare_track_par "compare_track_par"(track_par * t1, track_par * t2)
    
    sequence_par * c_read_sequence_par "read_sequence_par"(char * filename, int num_cams)
    sequence_par * c_new_sequence_par "new_sequence_par"(int num_cams)
    void c_free_sequence_par "free_sequence_par"(sequence_par * sp)
    int c_compare_sequence_par "compare_sequence_par"(sequence_par * sp1, sequence_par * sp2)
    
    volume_par * c_read_volume_par "read_volume_par"(char * filename);
    int c_compare_volume_par "compare_volume_par"(volume_par * v1, volume_par * v2);
    
    control_par * c_read_control_par "read_control_par"(char * filename);
    control_par * c_new_control_par "new_control_par"(int cams);
    void c_free_control_par "free_control_par"(control_par * cp);
    int c_compare_control_par "compare_control_par"(control_par * c1, control_par * c2);
    
    target_par* read_target_par(char *filename)

cdef numpy.ndarray wrap_1d_c_arr_as_ndarray(object base_obj, 
    int arr_size, int num_type, void * data, int copy):
    """
    Returns as NumPy array a value internally represented as a C array.
    
    Arguments:
    object base_obj - the object supplying the memory. Needed for refcounting.
    int arr_size - length of the C array.
    int num_type - type number as required by PyArray_SimpleNewFromData
    void* data - the underlying C array.
    int copy - 0 to return an ndarray whose data is the original data,
        1 to return a copy of the data.
    
    Returns:
    ndarray of length arr_size
    """
    cdef:
        numpy.npy_intp shape[1]  # for 1 dimensional array
        numpy.ndarray ndarr
    shape[0] = <numpy.npy_intp> arr_size
    
    ndarr = numpy.PyArray_SimpleNewFromData(1, shape, num_type, data)
    ndarr.base = <PyObject *> base_obj
    Py_INCREF(base_obj)
    
    if copy:
        return numpy.copy(ndarr)
    
    return ndarr
    
cdef class MultimediaParams:
    """
    Relates to photographing through several transparent media (air, tank 
    wall, water, etc.). Holds parameters related to media thickness and 
    refractive index.
    """

    def __init__(self, **kwargs):
        """
        Arguments (all optional):
        nlay - number of layers (default 1).
        n1 - index of refraction of first medium (usually air, 1).
        n2 - array, refr. indices of all but first and last layer.
        n3 - index of refraction of final medium (e.g. water, 1.33).
        d - thickness of all but first and last layers, [mm].
        """
        self._mm_np = < mm_np *> malloc(sizeof(mm_np))
        
        if kwargs.has_key('n1') and kwargs['n1'] is not None:
            self.set_n1(kwargs['n1'])
        if kwargs.has_key('n2') and kwargs.has_key('d') \
            and kwargs['n2'] is not None and kwargs['d'] is not None:
            self.set_layers(kwargs['n2'], kwargs['d'])
        if kwargs.has_key('n3') and kwargs['n3'] is not None:
            self.set_n3(kwargs['n3'])
             
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
        get the mid-layer refractive indices (n2) as a numpy array
        
        Arguments: 
        copy - False for returned numpy object to take ownership of 
            the memory of the n2 field of the underlying mm_np struct. True
            (default) to return a COPY instead.
        """
        arr_size = sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0])
        return wrap_1d_c_arr_as_ndarray(self, arr_size, numpy.NPY_DOUBLE, 
            self._mm_np[0].n2, copy)
    
    def get_d(self, copy=True):
        arr_size = sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0])
        return wrap_1d_c_arr_as_ndarray(self, arr_size , numpy.NPY_DOUBLE, self._mm_np[0].d, copy)
           
    def get_n3(self):
        return self._mm_np[0].n3
    
    def set_n3(self, n3):
        self._mm_np[0].n3 = n3

    def __richcmp__(MultimediaParams self, MultimediaParams other, operator):
        c_compare_result = c_compare_mm_np(self._mm_np, other._mm_np)
        if (operator == 2):  # "==" action was performed
            return (c_compare_result != 0)
        elif(operator == 3):  # "!=" action was performed
                return (c_compare_result == 0)
        else: raise TypeError("Unhandled comparison operator " + operator)
        
    def __str__(self):
        n2_str = "{"
        for i in range(sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0]) - 1):
            n2_str = n2_str + str(self._mm_np[0].n2[i]) + ", "
        n2_str += str(self._mm_np[0].n2[i + 1]) + "}"
        
        d_str = "{"
        for i in range(sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0]) - 1) :
            d_str += str(self._mm_np[0].d[i]) + ", "
            
        d_str += str(self._mm_np[0].d[i + 1]) + "}"
        
        return "nlay=\t{} \nn1=\t{} \nn2=\t{} \nd=\t{} \nn3=\t{} ".format(
                str(self._mm_np[0].nlay),
                str(self._mm_np[0].n1),
                n2_str,
                d_str,
                str(self._mm_np[0].n3))
        
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
      
    def read_track_par(self, file_name):
        """
        Reads tracking parameters from an old-style .par file having the
        objects' arguments ordered one per line.
        """
        free(self._track_par)
        self._track_par = c_read_track_par(file_name)
    
    # Checks for equality between this and other trackParams objects
    # Gives the ability to use "==" and "!=" operators on two trackParams objects
    def __richcmp__(TrackingParams self, TrackingParams other, operator):
        c_compare_result = c_compare_track_par(self._track_par, other._track_par)
        if (operator == 2):  # "==" action was performed
            return (c_compare_result != 0)
        elif(operator == 3):  # "!=" action was performed
                return (c_compare_result == 0)
        else: raise TypeError("Unhandled comparison operator " + operator)
             
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
        num_cams - number of camras used in the scene.
        image_base - a list of image base names, to which the frame number 
            is added during sequence operations.
        frame_range - (first, last)
        """
        if 'num_cams' in kwargs:
            num_cams = kwargs['num_cams']
        elif 'image_base' in kwargs:
            num_cams = len(kwargs['image_base'])
        else:
            raise ValueError(
                "SequenceParams requires either num_cams or image_base")
        
        self._sequence_par = c_new_sequence_par(num_cams)
        
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
        
    def read_sequence_par(self, filename, num_cams):
        """
        Reads sequence parameters from a config file with the following format:
        each line is a value, first num_cams values are image names, 
        (num_cams+1)th is the first number in the sequence, (num_cams+2)th line
        is the last value in the sequence.
        
        Arguments:
        filename - path to the text file containing the parameters.
        num_cams - expected number of cameras
        """
        # free the memory of previous C struct and its inner strings before 
        # creating a new one.
        c_free_sequence_par(self._sequence_par)
        
        # read parameters from file to a new sequence_par C struct
        self._sequence_par = c_read_sequence_par(filename, num_cams)
        
    # Get image base name of camera #cam
    def get_img_base_name(self, cam):
        cdef char * c_str = self._sequence_par[0].img_base_name[cam]
        cdef py_str = c_str
        return py_str
    
    # Set image base name for camera #cam
    def set_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char * c_string = py_byte_string
        strncpy(self._sequence_par[0].img_base_name[cam], c_string, len(new_img_name) + 1)
    
    # Checks for equality between this and other SequenceParams objects
    # Gives the ability to use "==" and "!=" operators on two SequenceParams objects
    def __richcmp__(SequenceParams self, SequenceParams other, operator):
        c_compare_result = c_compare_sequence_par(self._sequence_par, other._sequence_par)
        if (operator == 2):  # "==" action was performed
            return (c_compare_result != 0)
        elif(operator == 3):  # "!=" action was performed
                return (c_compare_result == 0)
        else: raise TypeError("Unhandled comparison operator " + operator)
        
    def __dealloc__(self):
        c_free_sequence_par(self._sequence_par)
        
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
        arr_size = sizeof(self._volume_par[0].X_lay) / sizeof(self._volume_par[0].X_lay[0])
        return wrap_1d_c_arr_as_ndarray(self, arr_size , numpy.NPY_DOUBLE, self._volume_par[0].X_lay, copy)
    
    def set_X_lay(self, X_lay):
        for i in range(len(X_lay)):
            self._volume_par[0].X_lay[i] = X_lay[i]
            
    def get_Zmin_lay(self, copy=True):
        arr_size = sizeof(self._volume_par[0].Zmin_lay) / sizeof(self._volume_par[0].Zmin_lay[0])
        return wrap_1d_c_arr_as_ndarray(self, arr_size , numpy.NPY_DOUBLE, self._volume_par[0].Zmin_lay, copy)

    def set_Zmin_lay(self, Zmin_lay):
        for i in range(len(Zmin_lay)):
            self._volume_par[0].Zmin_lay[i] = Zmin_lay[i]
            
    def get_Zmax_lay(self, copy=True):
        arr_size = sizeof(self._volume_par[0].Zmax_lay) / sizeof(self._volume_par[0].Zmax_lay[0])
        return wrap_1d_c_arr_as_ndarray(self, arr_size , numpy.NPY_DOUBLE, self._volume_par[0].Zmax_lay, copy)
    
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
        
    def read_volume_par(self, filename):
        """
        read_volume_par() reads parameters of illuminated volume from a config 
        file with the following format: each line is a value, in this order:
        
        1. X_lay[0]
        2. Zmin_lay[0]
        3. Zmax_lay[0]
        4. X_lay[1]
        5. Zmin_lay[1]
        6. Zmax_lay[1]
        7. cnx
        8. cny
        9. cn
        10.csumg
        11.corrmin
        12.eps0
        
        Argument:
        filename - path to the text file containing the parameters.
        """
        # free the memory of previous C struct 
        free(self._volume_par)
        # read parameters from file to a new volume_par C struct
        self._volume_par = c_read_volume_par(filename)
    
    # Checks for equality between self and other VolumeParams objects
    # Gives the ability to use "==" and "!=" operators on two VolumeParams objects
    def __richcmp__(VolumeParams self, VolumeParams other, operator):
        c_compare_result = c_compare_volume_par(self._volume_par, other._volume_par)
        if (operator == 2):  # "==" action was performed
            return (c_compare_result != 0)
        elif(operator == 3):  # "!=" action was performed
                return (c_compare_result == 0)
        else: raise TypeError("Unhandled comparison operator " + operator)
        
    def __dealloc__(self):
        free(self._volume_par)
        
cdef class ControlParams:
    """
    Wrapping the control_par C struct (declared in liboptv/paramethers.h) for 
    pythonic access. Objects of this type can be checked for equality using 
    "==" and "!=" operators.
    """
    def __init__(self, cams, flags=[], image_size=None, pixel_size=None,
        cam_side_n=None, wall_ns=None, wall_thicks=None, object_side_n=None):
        """
        Arguments (all optional except num_cams):
        num_cams - number of camras used in the scene.
        flags - a list containings name of set flags, select from 'hp', 
            'allcam', 'headers'.
        image_size - sequence, w,h image size in pixels.
        pixel_size - sequence, w,h pixel size in mm.
        cam_side_n, wall_ns, wall_thicks, object_side_n - see MultimediaParams
        """
        self._control_par = c_new_control_par(cams)
        self.set_hp_flag('hp' in flags)
        self.set_allCam_flag('allcam' in flags)
        self.set_tiff_flag('headers' in flags)
        self.set_chfield(0) # legacy stuff.
        
        if image_size is not None:
            self.set_image_size(image_size)
        if pixel_size is not None:
            self.set_pixel_size(pixel_size)
        
        self._multimedia_params = MultimediaParams(
            n1=cam_side_n, n2=wall_ns, d=wall_thicks, n3=object_side_n)
        free(self._control_par[0].mm)
        self._control_par[0].mm = self._multimedia_params._mm_np
        
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
    def read_control_par(self, filename):
        # free the memory of previous C struct and its inner strings before creating a new one
        c_free_control_par(self._control_par)
        # memory for _multimedia_params' mm_np struct was just freed,
        # so set _multimedia params' mm_np pointer to NULL to avoid double free corruption 
        self._multimedia_params._mm_np = NULL
        # read parameters from file to a new control_par C struct
        self._control_par = c_read_control_par(filename)
        # set _multimedia_params's _mm_np to the new mm_np that just allocated and read into _control_par
        self._multimedia_params.set_mm_np(self._control_par[0].mm)
        
    # Get image base name of camera #cam
    def get_img_base_name(self, cam):
        cdef char * c_str = self._control_par[0].img_base_name[cam]
        cdef py_str = c_str
        return py_str
    
    # Set image base name for camera #cam
    def set_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char * c_string = py_byte_string
        strncpy(self._control_par[0].img_base_name[cam], c_string, len(new_img_name) + 1)
    
    # Get calibration image base name of camera #cam
    def get_cal_img_base_name(self, cam):
        cdef char * c_str = self._control_par[0].cal_img_base_name[cam]
        cdef py_str = c_str
        return py_str
    
    # Set calibration image base name for camera #cam
    def set_cal_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char * c_string = py_byte_string
        strncpy(self._control_par[0].cal_img_base_name[cam], c_string, len(new_img_name) + 1)
    
    def get_multimedia_params(self):
        return self._multimedia_params
    
    # Checks for equality between this and other ControlParams objects
    # Gives the ability to use "==" and "!=" operators on two ControlParams objects
    def __richcmp__(ControlParams self, ControlParams other, operator):
        c_compare_result = c_compare_control_par(self._control_par, other._control_par)
        if (operator == 2):  # "==" action was performed
            return (c_compare_result != 0)
        elif(operator == 3):  # "!=" action was performed
                return (c_compare_result == 0)
        else: raise TypeError("Unhandled comparison operator " + operator)
        
    def __dealloc__(self):
        # set the mm pointer to NULL in order to prevent c_free_control_par 
        # function from freeing it. MultimediaParams object will free this
        # memory in its python destructor when there will be no references to it.
        
        self._control_par[0].mm = NULL
        c_free_control_par(self._control_par)

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
        return wrap_1d_c_arr_as_ndarray(self, num_cams, numpy.NPY_INT, 
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
    
    def read(self, inp_filename):
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
        free(self._targ_par)
        self._targ_par = read_target_par(inp_filename)
        
        if self._targ_par == NULL:
            raise IOError("Problem reading target recognition parameters.")
    
    def __dealloc__(self):
        free(self._targ_par)
