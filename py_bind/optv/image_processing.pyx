from optv.parameters cimport ControlParams, control_par
import numpy as np
from twisted.cred import error
cimport numpy as np

def preprocess_image(np.ndarray[ndim=2, dtype=np.uint8_t] input,
                   int filter_hp,
                   ControlParams control,
                   int lowpass_dim=0,
                   filter_file=None,
                   np.ndarray[ndim=2, dtype=np.uint8_t] output=None):
    '''
    preprocess_image() - perform the steps necessary for preparing an image to 
    particle detection: an averaging (smoothing) filter on an image, optionally
    followed by additional user-defined filter.
    
    Arguments:
    numpy.ndarray input - numpy 2d array representing the source image to filter.
    int filter_hp - flag for additional filtering of _hp. 1 for lowpass, 2 for 
        general 3x3 filter given in parameter ``filter_file``.
    ControlParams control - image details such as size and image half for 
    interlaced cases.
    int lowpass_dim - dimension of subtracted lowpass image.
    filter_file - path to a text file containing the filter matrix to be
        used in case ```filter_hp == 2```. One line per row, white-space 
        separated columns.
    numpy.ndarray output - result numpy 2d array representing the source 
        image to filter. Same size as img.
    
    Returns:
    numpy.ndarray representing the result image.
    '''

    # check arrays dimensions
    if input.ndim != 2:
        raise TypeError("Input matrix must be two-dimensional")
    if output != None and (input.shape[0] != output.shape[0] or
                         input.shape[1] != output.shape[1]):
        raise ValueError("Different shapes of input and output images.")
    else:
        output = np.empty_like(input)
    
    if filter_hp == 2:
        if filter_file == None or not isinstance(filter_file, basestring):
            raise ValueError("Expecting a filter file name, received None or non-string.")
    else:
        filter_file=""
        
    for arr in (input, output):
        if not arr.flags['C_CONTIGUOUS']:
            np.ascontiguousarray(arr)
    
    if (1 != prepare_image( < unsigned char *> input.data,
                            < unsigned char *> output.data,
                            lowpass_dim,
                            filter_hp,
                            filter_file,
                            control._control_par)):
        raise Exception("prepare_image C function failed: "
                      + "failure of memory allocation or filter file reading")
    return output              
