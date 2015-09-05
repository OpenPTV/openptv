#Implementation of Python binding to parameters.h
from libc.stdlib cimport malloc, free
from libc.string cimport strncpy

import numpy

cdef extern from "optv/parameters.h":
    track_par * c_read_track_par "read_track_par"(char * file_name)
    int c_compare_track_par "compare_track_par"(track_par * t1, track_par * t2)
    
    sequence_par * c_read_sequence_par "read_sequence_par"(char * filename)
    sequence_par * c_get_new_sequence_par "get_new_sequence_par"()
    void c_free_sequence_par "free_sequence_par"(sequence_par * sp)
    int c_compare_sequence_par "compare_sequence_par"(sequence_par * sp1, sequence_par * sp2)
    
    volume_par * c_read_volume_par "read_volume_par"(char * filename);
    int c_compare_volume_par "compare_volume_par"(volume_par * v1, volume_par * v2);
    
cdef class MultimediaParams:

    def __init__(self, **kwargs):
        
        self._mm_np = <mm_np *>malloc(sizeof(mm_np))
        
        self.set_nlay(kwargs['nlay'])
        self.set_n1(kwargs['n1'])
        self.set_n2(kwargs['n2'])
        self.set_d(kwargs['d'])
        self.set_n3(kwargs['n3'])
        self.set_lut(kwargs['lut'])
    
    def get_nlay(self):
        return self._mm_np[0].nlay
    
    def set_nlay(self, nlay):
        self._mm_np[0].nlay = nlay
        
    def get_n1(self):
        return self._mm_np[0].n1
    
    def set_n1(self, n1):
        self._mm_np[0].n1 = n1
        
    def get_n2(self):#TODO return numpy
        arr_size = sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0])
        n2_np_arr = numpy.empty(arr_size)
        for i in range(len(n2_np_arr)):
            n2_np_arr[i] = self._mm_np[0].n2[i]
        return n2_np_arr
    
    def set_n2(self, n2):
        for i in range(len(n2)):
            self._mm_np[0].n2[i] = n2[i]
            
    def get_d(self):
        arr_size = sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0])
        d_np_arr = numpy.empty(arr_size)
        
        for i in range(len(d_np_arr)):
            d_np_arr[i] = self._mm_np[0].d[i]
        return d_np_arr
        
    def set_d(self, d):
        for i in range(len(d)):
            self._mm_np[0].d[i] = d[i]
        
    def get_n3(self):
        return self._mm_np[0].n3
    
    def set_n3(self, n3):
        self._mm_np[0].n3 = n3
        
    def get_lut(self):
        return self._mm_np[0].lut
    
    def set_lut(self, lut):
        self._mm_np[0].lut = lut
        
    def __str__(self):
        n2_str="{"
        for i in range(sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0]) -1 ):
            n2_str = n2_str+ str(self._mm_np[0].n2[i]) + ", "
        n2_str += str(self._mm_np[0].n2[i+1]) + "}"
        
        d_str="{"
        for i in range(sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0]) -1 ) :
            d_str += str(self._mm_np[0].d[i]) + ", "
            
        d_str += str(self._mm_np[0].d[i+1]) + "}"
        
        return "nlay=\t{} \nn1=\t{} \nn2=\t{} \nd=\t{} \nn3=\t{} \nlut=\t{} ".format(
                str(self._mm_np[0].nlay),
                str(self._mm_np[0].n1),
                n2_str,
                d_str,
                str(self._mm_np[0].n3),
                str(self._mm_np[0].lut))
        
        def __dealloc__(self):
            free(self._mm_np)

# Wrapping the track_par C struct for pythonic access
# Binding the read_track_par C function
# Objects of this type can be checked for equality using "==" and "!=" operators

cdef class TrackingParams:   
    def __init__(self, dacc, dangle, dvxmax, dvxmin,
                 dvymax, dvymin, dvzmax, dvzmin,
                 dsumg, dn, dnx, dny, add):
        
        self._track_par = < track_par *> malloc(sizeof(track_par))

        self.set_dacc(dacc)
        self.set_dangle(dangle)
        self.set_dvxmax(dvxmax)
        self.set_dvxmin(dvxmin)
        
        self.set_dvymax(dvymax)
        self.set_dvymin(dvymin)
        self.set_dvzmax(dvzmax)
        self.set_dvzmin(dvzmin)
        
        self.set_dsumg(dsumg)
        self.set_dn(dn)
        self.set_dnx(dnx)
        self.set_dny(dny)
        self.set_add(add)
        
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
        return self._track_par[0].dangle
    
    def set_dangle(self, dangle):
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
    def __init__(self):
        self._sequence_par = c_get_new_sequence_par()
        
    def get_first(self):
        return self._sequence_par[0].first
    
    def set_first(self, first):
        self._sequence_par[0].first = first
        
    def get_last(self):
        return self._sequence_par[0].last
    
    def set_last(self, last):
        self._sequence_par[0].last = last
        
    # Reads sequence parameters from a config file with the
    # following format: each line is a value, first 4 values are image names,
    # 5th is the first number in the sequence, 6th line is the last value in the
    # sequence.
    # 
    # Argument:
    # filename - path to the text file containing the parameters.
    def read_sequence_par(self, filename):
        # free the memory of previous C struct and its inner strings before creating a new one
        c_free_sequence_par(self._sequence_par)
        # read parameters from file to a new sequence_par C struct
        self._sequence_par = c_read_sequence_par(filename)
        
    # Get image base name of camera #cam
    def get_img_base_name(self, cam):
        cdef char * c_str = self._sequence_par[0].img_base_name[cam]
        cdef py_str = c_str
        return py_str
    
    # Set image base name for camera #cam
    def set_img_base_name(self, cam, str new_img_name):
        py_byte_string = new_img_name.encode('UTF-8')
        cdef char* c_string = py_byte_string
        strncpy(self._sequence_par[0].img_base_name[cam], c_string, len(new_img_name)+1)
    
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
        
# Wrapping the volume_par C struct (declared in liboptv/paramethers.h) for pythonic access
# Objects of this type can be checked for equality using "==" and "!=" operators
cdef class VolumeParams:
    def __init__(self):
        self._volume_par = <volume_par*>malloc(sizeof(volume_par))
    
    # Getters and setters
    def get_X_lay(self):
        arr_size = sizeof(self._volume_par[0].X_lay) / sizeof(self._volume_par[0].X_lay[0])
        ret_np_arr = numpy.empty(arr_size)
        for i in range(arr_size):
            ret_np_arr[i] = self._volume_par[0].X_lay[i]
        return ret_np_arr
    
    def set_X_lay(self, X_lay):
        for i in range(len(X_lay)):
            self._volume_par[0].X_lay[i] = X_lay[i]
            
    def get_Zmin_lay(self):
        arr_size = sizeof(self._volume_par[0].Zmin_lay) / sizeof(self._volume_par[0].Zmin_lay[0])
        ret_np_arr = numpy.empty(arr_size)
        for i in range(arr_size):
            ret_np_arr[i] = self._volume_par[0].Zmin_lay[i]
        return ret_np_arr
    
    def set_Zmin_lay(self, Zmin_lay):
        for i in range(len(Zmin_lay)):
            self._volume_par[0].Zmin_lay[i] = Zmin_lay[i]
            
    def get_Zmax_lay(self):
        arr_size = sizeof(self._volume_par[0].Zmax_lay) / sizeof(self._volume_par[0].Zmax_lay[0])
        ret_np_arr = numpy.empty(arr_size)
        for i in range(arr_size):
            ret_np_arr[i] = self._volume_par[0].Zmax_lay[i]
        return ret_np_arr
    
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
        
    # read_volume_par() reads parameters of illuminated volume from a config file
    # with the following format: each line is a value, in this order:
    # 1. X_lay[0]
    # 2. Zmin_lay[0]
    # 3. Zmax_lay[0]
    # 4. X_lay[1]
    # 5. Zmin_lay[1]
    # 6. Zmax_lay[1]
    # 7. cnx
    # 8. cny
    # 9. cn
    # 10.csumg
    # 11.corrmin
    # 12.eps0
    #
    # Argument:
    # filename - path to the text file containing the parameters.
    def read_volume_par(self, filename):
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
        

