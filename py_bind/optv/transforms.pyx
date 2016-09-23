import numpy as np
cimport numpy as np
from optv.parameters cimport ControlParams
from optv.calibration cimport Calibration

def check_inputs(inp_arr, out_arr):
    """
    All functions here have the same behaviour of getting an (n,2) array and
    returning an array of the same shape, possibly the same one given as output
    parameter. This funcion checks that the rules of input are adhered to, 
    raises the appropriate exception if not, and returns the appropriate 
    (possibly new) output array to work with.
    """
    # Raise exception if received non Nx2 shaped ndarray
    # or if output and input arrays' shapes do not match.
    if inp_arr.shape[1] != 2:
        raise TypeError("Only two-column arrays accepted for conversion.")
        
    if out_arr is None:
        out_arr = np.empty_like(inp_arr)
    else:
        if np.any(inp_arr.shape != out_arr.shape):
            raise TypeError("Unmatching shape of input and output arrays: " 
                            + str(inp_arr.shape) + " != " + str(out_arr.shape))
    return out_arr
    
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
    out = check_inputs(input, out)

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
    if no array was passed for output - returns a new numpy ndarray with 
    converted coordinates
    '''
    return brown_affine_generic(input, calibration._calibration.added_par, out,
        correct_brown_affin)
  
def distort_arr_brown_affine(np.ndarray[ndim=2, dtype=np.float_t] input,
                                Calibration calibration,
                                np.ndarray[ndim=2, dtype=np.float_t] out=None):
    '''
    Transformation with Brown + affine. 
    input - input Numpy ndarray of Nx2 shape.
    calibration - Calibration object that holds parameters needed for transformation.
    out - OPTIONAL Numpy ndarray, same shape as input.
    
    Returns:
    if no array was passed for output - returns a new numpy ndarray with converted coordinates
    '''
    return brown_affine_generic(input, calibration._calibration.added_par, out,
        distort_brown_affin)

cdef brown_affine_generic(np.ndarray[ndim=2, dtype=np.float_t] input,
                        ap_52 c_ap_52,
                        np.ndarray[ndim=2, dtype=np.float_t] out,
                        void affine_function(double, double, ap_52 , double * , double *)):
    out = check_inputs(input, out)
    
    for i in range(input.shape[0]):
        affine_function((< double *> np.PyArray_GETPTR2(input, i, 0))[0]
                        , (< double *> np.PyArray_GETPTR2(input, i, 1))[0]
                        , c_ap_52
                        , < double *> np.PyArray_GETPTR2(out, i, 0)
                        , < double *> np.PyArray_GETPTR2(out, i, 1))
    return out

def distorted_to_flat(np.ndarray[ndim=2, dtype=np.float_t] inp,
    Calibration calibration, np.ndarray[ndim=2, dtype=np.float_t] out=None,
    double tol=0.00001):
    """
    Full, exact conversion of distorted metric coordinates to flat unshifted 
    metric coordinates. By 'exact' it is meant that the correction iteration 
    is allowed to do more than one step, with given tolerance.
    
    Arguments:
    input - input (n,2) array, distorted metric coordinates.
    calibration - Calibration object that holds parameters needed for 
        transformation.
    out - (optional) ndarray, same shape as input. If given, result is placed
        in the memory belonging to this array.
    tol - (optional) tolerance of improvement in predicting radial position
        between iterations of the correction loop. There is a sensible default
        but if you need more or less accuracy you can change that.
    
    Returns:
    the out array with metric flat unshifted coordinates, or a new array of the
    correct size with the same results.
    """
    out = check_inputs(inp, out)
    
    for pt_num, pt in enumerate(inp):
        dist_to_flat(pt[0], pt[1], calibration._calibration, 
            <double *> np.PyArray_GETPTR2(out, pt_num, 0),
            <double *> np.PyArray_GETPTR2(out, pt_num, 1), tol)
    
    return out