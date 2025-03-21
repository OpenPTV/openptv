# cython: language_level=3
# distutils: language = c

# Cython definitions for parameters.h
cdef extern from "optv/parameters.h":
    ctypedef struct mm_np:
        int nlay
        double n1
        double n2[3]
        double d[3]
        double n3
    
    ctypedef struct track_par:
        double dacc, dangle, dvxmax, dvxmin
        double dvymax, dvymin, dvzmax, dvzmin
        int dsumg, dn, dnx, dny, add
    
    ctypedef struct sequence_par:
        char ** img_base_name
        int first, last
        
    ctypedef struct volume_par:
        double X_lay[2]
        double Zmin_lay[2]
        double Zmax_lay[2]
        double cn, cnx, cny, csumg, eps0, corrmin
        
    ctypedef struct control_par:
        int num_cams
        char **img_base_name
        char **cal_img_base_name
        int hp_flag
        int allCam_flag
        int tiff_flag
        int imx
        int imy
        double pix_x
        double pix_y
        int chfield
        mm_np *mm
    
    ctypedef struct target_par:
        int discont
        int gvthres[4]    # grey value threshold per camera.
        int nnmin, nnmax  # bounds for number of pixels in target.
        int nxmin, nxmax  # same in x dimension.
        int nymin, nymax  # same in y dimension.
        int sumg_min      # minimal sum of grey values in target.
        int cr_sz         # correspondence parameter.
    
    target_par* read_target_par(char* filename)
    control_par* read_control_par(char* filename)

    # Add these function declarations
    track_par* read_track_par(char* filename)
    sequence_par* read_sequence_par(char* filename, int num_cams)
    volume_par* read_volume_par(char* filename)

    # Add these free function declarations
    void free_control_par(control_par *cp)
    void free_sequence_par(sequence_par *sp)

    # Add comparison function declarations
    int compare_control_par(control_par *c1, control_par *c2)
    int compare_sequence_par(sequence_par *sp1, sequence_par *sp2)
    int compare_track_par(track_par *t1, track_par *t2)
    int compare_volume_par(volume_par *v1, volume_par *v2)
    int compare_mm_np(mm_np *mm1, mm_np *mm2)

    # Add new_control_par declaration
    control_par* new_control_par(int num_cams)

    sequence_par* new_sequence_par(int num_cams)
    void free_sequence_par(sequence_par* sp)

cdef class MultimediaParams:
    cdef mm_np* _mm_np
    cdef void set_mm_np(MultimediaParams self, mm_np * other_mm_np_c_struct)
    
cdef class TrackingParams:
    cdef track_par* _track_par
    cdef bytes _filename_bytes

cdef class SequenceParams:
    cdef sequence_par* _sequence_par
    cdef bytes _filename_bytes

cdef class VolumeParams:
    cdef volume_par *_volume_par
    cdef bytes _filename_bytes

cdef class ControlParams:
    cdef control_par* _control_par
    cdef MultimediaParams _multimedia_params
    cdef bytes _filename_bytes

cdef class TargetParams:
    cdef target_par* _targ_par
    cdef bytes _filename_bytes  # Add storage for bytes objects
