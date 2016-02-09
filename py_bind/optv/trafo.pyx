import numpy as np
cimport numpy as np
from optv.parameters cimport ControlParams

def convert_pixel_to_metric(input, ControlParams control_par, out=None):
    if type(input) != np.ndarray or (out != None and type(out) != np.ndarray):
        raise TypeError("Unexpected object type received as array (" + type(input)
                        + "). Only numpy ndarray is accepted.")
    shape_tpl = input.shape
    if len(shape_tpl) != 2 or shape_tpl[1] != 2:
        raise TypeError("Unexpected shape of input array: " 
                        + str(input.shape) + ". Only (n,2) shape is accepted.")
     
    if out == None:
        out = np.empty(shape_tpl)    
    
    if out.shape != shape_tpl or len(input) != len(out):
        raise TypeError("Arrays of different shape or/and size received for input and output.")     
        
    cdef double x, y
    for i in range(len(input)):
        pixel_to_metric(& x, & y , input[i][0], input[i][1] , control_par._control_par)
        out[i][0] = x
        out[i][1] = y
                
    return out
