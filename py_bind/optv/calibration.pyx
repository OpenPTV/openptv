# Implementation of Python binding to calibrtion.h C module
from libc.stdlib cimport malloc, free
import numpy
from optv.calibration import Calibration
from mhlib import isnumeric

cdef extern from "optv/calibration.h":
    calibration *read_calibration(char *ori_file, char *add_file,
        char *fallback_file)
    int write_calibration(calibration *cal, char *filename, char *add_file)
    void rotation_matrix(Exterior *ex)
    
cdef class Calibration:
    def __init__(self):
        self._calibration = <calibration *> malloc(sizeof(calibration))
        
    def from_file(self, ori_file=None, add_file=None, fallback_file=None):
        self._calibration = read_calibration(
            (<char *>ori_file if ori_file != None else < char *> 0),
            (<char *>add_file if add_file != None else < char *> 0), NULL)
        
    def write(self, filename, add_file):
        success = write_calibration(self._calibration, filename, add_file)
        if not success:
            raise IOError("Failed to write calibration.")
    
    # Sets exterior position.
    # Parameter: x_y_z_np - numpy array of 3 elements for x, y, z
    def set_pos(self, x_y_z_np):
        if len(x_y_z_np) != 3:
             raise ValueError("Illegal array argument " + x_y_z_np.__str__() + \
                " for x, y, z. Expected array/list of 3 numbers")
        self._calibration[0].ext_par.x0 = x_y_z_np[0]
        self._calibration[0].ext_par.y0 = x_y_z_np[1]
        self._calibration[0].ext_par.z0 = x_y_z_np[2]
        
    # Returns numpy array of 3 elements representing exterior's x, y, z
    def get_pos(self):
        ret_x_y_z_np = numpy.empty(3)
        ret_x_y_z_np[0] = self._calibration[0].ext_par.x0
        ret_x_y_z_np[1] = self._calibration[0].ext_par.y0
        ret_x_y_z_np[2] = self._calibration[0].ext_par.z0
        
        return ret_x_y_z_np
        
    # Sets angles (omega, phi, kappa) and recalculates Dmatrix accordingly
    # Parameter o_p_k_np - numpy of 3 elements
    def set_angles(self, o_p_k_np):
        if len(o_p_k_np) != 3:
            raise ValueError("Illegal array argument " + o_p_k_np.__str__() + \
                " for omega, phi, kappa. Expected array/list of 3 numbers")
        self._calibration[0].ext_par.omega = o_p_k_np[0]
        self._calibration[0].ext_par.phi = o_p_k_np[1]
        self._calibration[0].ext_par.kappa = o_p_k_np[2]
        
        # recalculate the Dmatrix dm according to new angles
        rotation_matrix (&self._calibration[0].ext_par)
    
    # Returns a numpy array of 3 elements representing omega, phi, kappa
    def get_angles(self):
        ret_o_p_k_np = numpy.empty(3)
        ret_o_p_k_np[0] = self._calibration[0].ext_par.omega
        ret_o_p_k_np[1] = self._calibration[0].ext_par.phi
        ret_o_p_k_np[2] = self._calibration[0].ext_par.kappa
        
        return ret_o_p_k_np
    
    # Returns a 3x3 numpy array that represents Exterior's Dmatrix 
    def get_rotation_matrix(self):
        ret_dmatrix_np = numpy.empty(shape=(3, 3))
        for i in range(3):
            for j in range(3):
                ret_dmatrix_np[i][j] = self._calibration[0].ext_par.dm[i][j]
        
        return ret_dmatrix_np
    
    # Free memory
    def __dealloc__(self):
        free(self._calibration)

