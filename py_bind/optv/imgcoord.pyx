import numpy as np
cimport numpy as np

from optv.parameters cimport MultimediaParams, mm_np
from optv.calibration cimport Calibration, calibration

def flat_image_coord( np.ndarray[ndim=2, dtype=np.float_t] input,
               Calibration cal,
               MultimediaParams mult_params,
               np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
        #TODO documentation
    '''
    return image_coord_generic(input, out, cal._calibration, mult_params._mm_np, c_flat_image_coord)

def image_coord( np.ndarray[ndim=2, dtype=np.float_t] input,
               Calibration cal,
               MultimediaParams mult_params,
               np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
    #TODO documentation
    Transformation with Brown + affine. 
    input - input Numpy ndarray of Nx2 shape.
    control - Calibration object that holds parameters needed for transformation.
    out - OPTIONAL Numpy ndarray, same shape as input.
    
    Returns:
    if no array was passed for output - returns a new numpy ndarray with converted coordinates
    '''
    return image_coord_generic(input, out, cal._calibration, mult_params._mm_np, c_image_coord)

cdef image_coord_generic( np.ndarray[ndim=2, dtype=np.float_t] input,
                        np.ndarray[ndim=2, dtype=np.float_t] out,
                        calibration * c_cal,
                        mm_np * c_mm_np,
                        void image_coord_function(vec3d, calibration *, mm_np *, double *, double *)):
    # Raise exceptions if received non Nx3 shaped ndarray for input
    # or non Nx2 shaped ndarray for output
    # or if output and input arrays' number of rows do not match.
    if input.shape[1] != 3:
        raise TypeError("Input matrix must have three columns (each row for 3d coordinate).")
    if out != None:
        if out.shape[1] != 2:
            raise TypeError("Output matrix must have two columns (each row for 2d coordinate).")
        if input.shape[0] != out.shape[0]:
            raise TypeError("Unmatching number of rows in input and output arrays: (" 
                            + str(input.shape[0]) + "," + str(input.shape[1]) 
                            + ") != (" + str(out.shape[0]) + "," + str(out.shape[1]) + ")")
    else: # out==None
        # If there was no array passed for output :
        # create new array for output with same number of rows as input and with only 2 columns 
        out = np.empty((input.shape[0], 2))
        
    cdef vec3d temp3dvec
    
    for i in range(input.shape[0]):
        
        temp3dvec[0] = (< double *> np.PyArray_GETPTR2(input, i, 0))[0]
        temp3dvec[1] = (< double *> np.PyArray_GETPTR2(input, i, 1))[0]
        temp3dvec[2] = (< double *> np.PyArray_GETPTR2(input, i, 2))[0]
        
        image_coord_function(temp3dvec,
                           c_cal,
                           c_mm_np,
                           < double *> np.PyArray_GETPTR2(out, i, 0), 
                           < double *> np.PyArray_GETPTR2(out, i, 1))
    return out
