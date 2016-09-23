import numpy as np
cimport numpy as np

from optv.parameters cimport MultimediaParams, mm_np
from optv.calibration cimport Calibration, calibration

def flat_image_coordinates(np.ndarray[ndim=2, dtype=np.float_t] input,
                           Calibration cal,
                           MultimediaParams mult_params,
                           np.ndarray[ndim=2, dtype=np.float_t] output=None):
    '''
    calculates projection from coordinates in
    world space to metric coordinates in image space without 
    distortions
    
    Arguments:
    input - a numpy array of vectors of position in 3D (X,Y,Z real space)
    Calibration cal - parameters of the camera on which to project.
    MultimediaParams mult_params- layer thickness and refractive index parameters.
    output (optional) - a numpy array of input length containing x,y pixel 
        coordinates of projection in the image space. New array is created if
        ``output`` is ``None``. 
    '''
    check_arrays(input, output)
    
    # If no array was passed for output:
    # create new array with same number of rows as in input 
    # with only 2 columns    
    if output is None:        
        output = np.empty((input.shape[0], 2))
    input = np.ascontiguousarray(input)
     
    for i in range(input.shape[0]):
        flat_image_coord(<vec3d>np.PyArray_GETPTR2(input, i, 0),
                         cal._calibration,
                         mult_params._mm_np,
                         < double *> np.PyArray_GETPTR2(output, i, 0),
                         < double *> np.PyArray_GETPTR2(output, i, 1))
    return output

def image_coordinates(np.ndarray[ndim=2, dtype=np.float_t] input,
                      Calibration cal,
                      MultimediaParams mult_params,
                      np.ndarray[ndim=2, dtype=np.float_t] output=None):
    '''
    estimates metric coordinates in image space
    from the 3D position in the world and distorts it using the Brown 
    distortion model [1]
    
    Arguments:
    input - a numpy array of vectors of position in 3D (X,Y,Z real space)
    Calibration cal- parameters of the camera on which to project.
    MultimediaParams mult_params- layer thickness and refractive index parameters.
    output (optional) - a numpy array of input length containing x,y pixel coordinates of
    projection in the image space. New array is created if output=None. 
    '''
    check_arrays(input, output)
    
    # If no array was passed for output:
    # create new array with same number of rows as in input 
    # with only 2 columns    
    if output is None:        
        output = np.empty((input.shape[0], 2))
    input = np.ascontiguousarray(input)
     
    for i in range(input.shape[0]):
        img_coord(<vec3d>np.PyArray_GETPTR2(input, i, 0),
                         cal._calibration,
                         mult_params._mm_np,
                         < double *> np.PyArray_GETPTR2(output, i, 0),
                         < double *> np.PyArray_GETPTR2(output, i, 1))
    return output
    
def check_arrays(np.ndarray[ndim=2, dtype=np.float_t] input,
                 np.ndarray[ndim=2, dtype=np.float_t] output):
    # Raise exceptions if received non Nx3 shaped ndarray for input
    # or non Nx2 shaped ndarray for output
    # or if output and input arrays' number of rows do not match.
    if input.shape[1] != 3:
        raise TypeError("Input matrix must have three columns (each row for 3d coordinate).")
    if output is not None:
        if output.shape[1] != 2:
            raise TypeError("Output matrix must have two columns (each row for 2d coordinate).")
        if input.shape[0] != output.shape[0]:
            raise TypeError("Unmatching number of rows in input and output arrays: (" 
                            + str(input.shape[0]) + "," + str(input.shape[1]) 
                            + ") != (" + str(output.shape[0]) + "," + str(output.shape[1]) + ")")
    
