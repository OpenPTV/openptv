from optv.parameters cimport ControlParams, control_par
import numpy as np
cimport numpy as np
from six import string_types

def preprocess_image(np.ndarray[ndim=2, dtype=np.uint8_t] input_img,
                   int filter_hp,
                   ControlParams control,
                   int lowpass_dim=1,
                   filter_file=None,
                   np.ndarray[ndim=2, dtype=np.uint8_t] output_img=None):
    '''
    preprocess_image() - perform the steps necessary for preparing an image to 
    particle detection: an averaging (smoothing) filter on an image, optionally
    followed by additional user-defined filter.
    
    Arguments:
    numpy.ndarray input_img - numpy 2d array representing the source image to filter.
    int filter_hp - flag for additional filtering of _hp. 1 for lowpass, 2 for 
        general 3x3 filter given in parameter ``filter_file``.
    ControlParams control - image details such as size and image half for 
    interlaced cases.
    int lowpass_dim - half-width of lowpass filter, see fast_box_blur()'s filt_span
      parameter.
    filter_file - path to a text file containing the filter matrix to be
        used in case ```filter_hp == 2```. One line per row, white-space 
        separated columns.
    numpy.ndarray output_img - result numpy 2d array representing the source 
        image to filter. Same size as img.
    
    Returns:
    numpy.ndarray representing the result image.
    '''

    # check arrays dimensions
    if input_img.ndim != 2:
        raise TypeError("Input array must be two-dimensional")
    if (output_img is not None) and (input_img.shape[0] != output_img.shape[0] or
                         input_img.shape[1] != output_img.shape[1]):
        raise ValueError("Different shapes of input and output images.")
    else:
        output_img = np.empty_like(input_img)
    
    if filter_hp == 2:
        if filter_file == None or not isinstance(filter_file, string_types):
            raise ValueError("Expecting a filter file name, received None or non-string.")
    else:
        filter_file=b""
        
    for arr in (input_img, output_img):
        if not arr.flags['C_CONTIGUOUS']:
            np.ascontiguousarray(arr)
    
    if (1 != prepare_image( < unsigned char *> input_img.data,
                            < unsigned char *> output_img.data,
                            lowpass_dim,
                            filter_hp,
                            filter_file,
                            control._control_par)):
        raise Exception("prepare_image C function failed: "
                      + "failure of memory allocation or filter file reading")
    return output_img              
