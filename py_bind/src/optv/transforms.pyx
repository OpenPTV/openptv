import numpy as np
cimport numpy as np
from optv.parameters cimport control_par, ControlParams
from optv.calibration cimport calibration, Calibration, ap_52

# Define function pointer types - removed 'nogil' from typedef and made them noexcept
ctypedef void (*convert_func)(double*, double*, double, double, control_par*) noexcept
ctypedef void (*affine_func)(double, double, ap_52, double*, double*) noexcept

def check_inputs(double[:, ::1] inp_arr, double[:, ::1] out_arr=None):
    if inp_arr.shape[1] != 2:
        raise TypeError("Only two-column arrays accepted for conversion.")
        
    if out_arr is None:
        out_arr = np.empty_like(np.asarray(inp_arr))
    else:
        if inp_arr.shape[0] != out_arr.shape[0] or inp_arr.shape[1] != out_arr.shape[1]:
            raise TypeError("Unmatching shape of input and output arrays: " 
                          + str(inp_arr.shape) + " != " + str(out_arr.shape))
    return out_arr

cdef double[:, ::1] convert_generic(
        double[:, ::1] input,
        control_par* c_control,
        double[:, ::1] out,
        convert_func convert_function) noexcept:
    
    cdef:
        Py_ssize_t i
        double* out_ptr
        double* in_ptr
    
    for i in range(input.shape[0]):
        convert_function(
            &out[i, 0],
            &out[i, 1],
            input[i, 0],
            input[i, 1],
            c_control)
    return out

cdef double[:, ::1] brown_affine_generic(
        double[:, ::1] input,
        ap_52 c_ap_52,
        double[:, ::1] out,
        affine_func affine_function) noexcept:
    
    cdef:
        Py_ssize_t i
    
    for i in range(input.shape[0]):
        affine_function(
            input[i, 0],
            input[i, 1],
            c_ap_52,
            &out[i, 0],
            &out[i, 1])
    return out

def convert_arr_pixel_to_metric(np.ndarray[double, ndim=2] input,
                              ControlParams control,
                              np.ndarray[double, ndim=2] out=None):
    cdef:
        double[:, ::1] input_view = input
        double[:, ::1] out_view
        convert_func func = &pixel_to_metric
    
    out_view = check_inputs(input_view, None if out is None else out)
    out_view = convert_generic(input_view, control._control_par, out_view, func)
    return np.asarray(out_view)

def convert_arr_metric_to_pixel(np.ndarray[double, ndim=2] input,
                              ControlParams control,
                              np.ndarray[double, ndim=2] out=None):
    cdef:
        double[:, ::1] input_view = input
        double[:, ::1] out_view
        convert_func func = &metric_to_pixel
    
    out_view = check_inputs(input_view, None if out is None else out)
    out_view = convert_generic(input_view, control._control_par, out_view, func)
    return np.asarray(out_view)

def correct_arr_brown_affine(np.ndarray[double, ndim=2] input,
                           Calibration calibration,
                           np.ndarray[double, ndim=2] out=None):
    cdef:
        double[:, ::1] input_view = input
        double[:, ::1] out_view
        affine_func func = &correct_brown_affin
    
    out_view = check_inputs(input_view, None if out is None else out)
    out_view = brown_affine_generic(input_view, calibration._calibration.added_par, out_view, func)
    return np.asarray(out_view)

def distort_arr_brown_affine(np.ndarray[double, ndim=2] input,
                           Calibration calibration,
                           np.ndarray[double, ndim=2] out=None):
    cdef:
        double[:, ::1] input_view = input
        double[:, ::1] out_view
        affine_func func = &distort_brown_affin
    
    out_view = check_inputs(input_view, None if out is None else out)
    out_view = brown_affine_generic(input_view, calibration._calibration.added_par, out_view, func)
    return np.asarray(out_view)

def distorted_to_flat(np.ndarray[ndim=2, dtype=np.float64_t] inp,
    Calibration calibration, np.ndarray[ndim=2, dtype=np.float64_t] out=None,
    double tol=0.00001):
    """
    Full, exact conversion of distorted metric coordinates to flat unshifted 
    metric coordinates.
    
    Arguments:
    input - input (n,2) array, distorted metric coordinates.
    calibration - Calibration object that holds parameters needed for 
        transformation.
    out - (optional) ndarray, same shape as input. If given, result is placed
        in the memory belonging to this array.
    tol - (optional) tolerance of improvement in predicting radial position
        between iterations of the correction loop.
    
    Returns:
    the out array with metric flat unshifted coordinates, or a new array of the
    correct size with the same results.
    """
    
    if inp is None:
        raise ValueError("Input array cannot be None")
        
    if inp.shape[1] != 2:
        raise ValueError("Input array must have shape (n,2)")
        
    if out is not None and (out.shape != inp.shape):
        raise ValueError("Output array must have same shape as input array")
    
    if out is None:
        out = np.empty_like(inp)
    
    for pt_num, pt in enumerate(inp):
        dist_to_flat(pt[0], pt[1], calibration._calibration, 
            <double *> np.PyArray_GETPTR2(out, pt_num, 0),
            <double *> np.PyArray_GETPTR2(out, pt_num, 1), tol)
    
    return np.asarray(out)  # Ensure we return a numpy array, not a memoryview
