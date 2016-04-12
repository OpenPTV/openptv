import numpy as np
cimport numpy as np
from optv.parameters cimport ControlParams
from optv.calibration cimport Calibration

def convert_arr_pixel_to_metric(np.ndarray[ndim=2, dtype=np.float_t] input,
                                ControlParams control,
                                np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
    Convert NumPy 2d, Nx2 array from pixel coordinates to metric coordinates.
    Arguments:
    input-  input Numpy ndarray of Nx2 shape.
    control - ControlParams object that holds parameters needed for conversion.
    out - OPTIONAL numpy ndarray, same shape as input.
    
    Returns:
    if no array was passed for output returns a new numpy ndarray with converted coordinates
    '''
    return convert_generic(input, control._control_par, out, pixel_to_metric)
  
def convert_arr_metric_to_pixel(np.ndarray[ndim=2, dtype=np.float_t] input,
                                ControlParams control,
                                np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
    Convert NumPy 2d, Nx2 array from metric coordinates to pixel coordinates.
    input - input Numpy ndarray of Nx2 shape.
    control - ControlParams object that holds parameters needed for conversion.
    out - OPTIONAL Numpy ndarray, same shape as input.
    
    Returns:
    if no array was passed for output returns a new numpy ndarray with converted coordinates
    '''
    return convert_generic(input, control._control_par, out, metric_to_pixel)

cdef convert_generic(np.ndarray[ndim=2, dtype=np.float_t] input,
                        control_par * c_control,
                        np.ndarray[ndim=2, dtype=np.float_t] out,
                        void convert_function(double * , double * , double, double , control_par *)):
    # Raise exception if received non Nx2 shaped ndarray
    # or if output and input arrays' shapes do not match.
    if input.shape[1] != 2 or (out != None and out.shape[1] != 2):
        raise TypeError("Only two-column matrices accepted for conversion.")
    if out != None:
        if not(input.shape[0] == out.shape[0] and input.shape[1] == out.shape[1]):
            raise TypeError("Unmatching shape of input and output arrays: (" 
                            + str(input.shape[0]) + "," + str(input.shape[1]) 
                            + ") != (" + str(out.shape[0]) + "," + str(out.shape[1]) + ")")
    else:
        # If no array for output was passed (out==None):
        # create new array for output with  same shape as input array 
        out = np.empty_like(input)

    for i in range(input.shape[0]):
        convert_function(< double *> np.PyArray_GETPTR2(out, i, 0)
                        , < double *> np.PyArray_GETPTR2(out, i, 1)
                        , (< double *> np.PyArray_GETPTR2(input, i, 0))[0]
                        , (< double *> np.PyArray_GETPTR2(input, i, 1))[0]
                        , c_control)
    return out

# Affine #

def correct_arr_brown_affine(np.ndarray[ndim=2, dtype=np.float_t] input,
                                Calibration calibration,
                                np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
    Correct crd to geo with Brown + affine.
    input - input Numpy ndarray of Nx2 shape.
    control - Calibration object that holds parameters needed for transformation.
    out - OPTIONAL numpy ndarray, same shape as input.
    
    Returns:
    if no array was passed for output - returns a new numpy ndarray with converted coordinates
    '''
    return brown_affine_generic(input, calibration._calibration.added_par, out, correct_brown_affin)
  
def distort_arr_brown_affine(np.ndarray[ndim=2, dtype=np.float_t] input,
                                Calibration calibration,
                                np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
    Transformation with Brown + affine. 
    input - input Numpy ndarray of Nx2 shape.
    control - Calibration object that holds parameters needed for transformation.
    out - OPTIONAL Numpy ndarray, same shape as input.
    
    Returns:
    if no array was passed for output - returns a new numpy ndarray with converted coordinates
    '''
    return brown_affine_generic(input, calibration._calibration.added_par, out, distort_brown_affin)

cdef brown_affine_generic(np.ndarray[ndim=2, dtype=np.float_t] input,
                        ap_52 c_ap_52,
                        np.ndarray[ndim=2, dtype=np.float_t] out,
                        void affine_function(double, double, ap_52 , double * , double *)):
    # Raise exception if received non Nx2 shaped ndarray
    # or if output and input arrays' shapes do not match.
    if input.shape[1] != 2 or (out != None and out.shape[1] != 2):
        raise TypeError("Only two-column matrices accepted for conversion.")
    if out != None:
        if not(input.shape[0] == out.shape[0] and input.shape[1] == out.shape[1]):
            raise TypeError("Unmatching shape of input and output arrays: (" 
                            + str(input.shape[0]) + "," + str(input.shape[1]) 
                            + ") != (" + str(out.shape[0]) + "," + str(out.shape[1]) + ")")
    else:
        # If no array for output was passed (out==None):
        # create new array for output with  same shape as input array 
        out = np.empty_like(input)

    for i in range(input.shape[0]):
        affine_function((< double *> np.PyArray_GETPTR2(input, i, 0))[0]
                        , (< double *> np.PyArray_GETPTR2(input, i, 1))[0]
                        , c_ap_52
                        , < double *> np.PyArray_GETPTR2(out, i, 0)
                        , < double *> np.PyArray_GETPTR2(out, i, 1))
    return out
