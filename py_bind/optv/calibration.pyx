# Implementation of Python binding to calibrtion.h C module
#
# References:
# [1] Dracos, Th., ed.; Three Dimensional Velocity and Vorticity Measuring
#     and Image Analysis Techniques; 1996. Chapter 3.

from libc.stdlib cimport malloc, free
import numpy
cimport numpy as cnp

from optv.calibration import Calibration

cdef extern from "optv/calibration.h":
    calibration *read_calibration(char *ori_file, char *add_file,
        char *fallback_file)
    int write_calibration(calibration *cal, char *filename, char *add_file)
    void rotation_matrix(Exterior *ex)
    
cdef class Calibration:
    def __init__(self, pos=None, angs=None, prim_point=None, rad_dist=None,
        decent=None, affine=None, glass=None):
        """
        All arguments are optional arrays, default for all is zeros except 
        affine that defaults to [1, 0].
        
        Arguments:
        pos - camera external position (world position of promary point).
        angs - in radians CCW around the x,y,z axes respectively.
        prim_point - position of primary point rel. image plan. Object is 
            assumed to be in negative Z, and image plan in 0, so use positive
            Z coordinate.
        rad_dist - 3 radial distortion parameters, see [1].
        decent - 2 decentering parameters, see [1].
        affine - 2 parameters: scaling and x-shearing.
        glass - vector from world origin to glass, normal to glass.
        """
        self._calibration = <calibration *> malloc(sizeof(calibration))
        self._calibration.mmlut.data = NULL
        
        if pos is None:
            pos = numpy.zeros(3)
        if angs is None:
            angs = numpy.zeros(3)
        if prim_point is None:
            prim_point = numpy.zeros(3)
        if rad_dist is None:
            rad_dist = numpy.zeros(3)
        if decent is None:
            decent = numpy.zeros(2)
        if affine is None:
            affine = numpy.r_[1, 0]
        if glass is None:
            glass = numpy.zeros(3)
        
        self.set_pos(pos)
        self.set_angles(angs)
        self.set_primary_point(prim_point)
        self.set_radial_distortion(rad_dist)
        self.set_decentering(decent)
        self.set_affine_trans(affine)
        self.set_glass_vec(glass)
        
    def from_file(self, ori_file, add_file=None, fallback_file=None):
        """
        Populate calibration fields from .ori and .addpar files.
        
        Arguments:
        ori_file - path to file containing exterior, interior and glass
            parameters.
        add_file - optional path to file containing distortion parameters.
        fallback_file - optional path to file used in case ``add_file`` fails
            to open.
        """
        free(self._calibration);
        self._calibration = read_calibration(
            (<char *>ori_file if ori_file != None else < char *> 0),
            (<char *>add_file if add_file != None else < char *> 0), NULL)
        
    def write(self, filename, add_file):
        """
        Write the calibration data to disk. Uses two output file, one for the
        linear calibration part, and one for distortion parameters.
        
        Arguments:
        filename - path to file containing exterior, interior and glass
            parameters.
        add_file - optional path to file containing distortion parameters.
        """
        success = write_calibration(self._calibration, filename, add_file)
        if not success:
            raise IOError("Failed to write calibration.")
    
    def set_pos(self, x_y_z_np):
        """
        Sets exterior position.
        Parameter: x_y_z_np - numpy array of 3 elements for x, y, z
        """
        if len(x_y_z_np) != 3:
             raise ValueError("Illegal array argument " + x_y_z_np.__str__() + \
                " for x, y, z. Expected array/list of 3 numbers")
        self._calibration[0].ext_par.x0 = x_y_z_np[0]
        self._calibration[0].ext_par.y0 = x_y_z_np[1]
        self._calibration[0].ext_par.z0 = x_y_z_np[2]
        
    def get_pos(self):
        """
        Returns numpy array of 3 elements representing exterior's x, y, z
        """
        ret_x_y_z_np = numpy.empty(3)
        ret_x_y_z_np[0] = self._calibration[0].ext_par.x0
        ret_x_y_z_np[1] = self._calibration[0].ext_par.y0
        ret_x_y_z_np[2] = self._calibration[0].ext_par.z0
        
        return ret_x_y_z_np
        
    def set_angles(self, o_p_k_np):
        """
        Sets angles (omega, phi, kappa) and recalculates Dmatrix accordingly
        Parameter: o_p_k_np - array of 3 elements.
        """
        if len(o_p_k_np) != 3:
            raise ValueError("Illegal array argument " + o_p_k_np.__str__() + \
                " for omega, phi, kappa. Expected array/list of 3 numbers")
        self._calibration[0].ext_par.omega = o_p_k_np[0]
        self._calibration[0].ext_par.phi = o_p_k_np[1]
        self._calibration[0].ext_par.kappa = o_p_k_np[2]
        
        # recalculate the Dmatrix dm according to new angles
        rotation_matrix (&self._calibration[0].ext_par)
    
    def get_angles(self):
        """
        Returns a numpy array of 3 elements representing omega, phi, kappa
        """
        ret_o_p_k_np = numpy.empty(3)
        ret_o_p_k_np[0] = self._calibration[0].ext_par.omega
        ret_o_p_k_np[1] = self._calibration[0].ext_par.phi
        ret_o_p_k_np[2] = self._calibration[0].ext_par.kappa
        
        return ret_o_p_k_np
    
    def get_rotation_matrix(self):
        """
        Returns a 3x3 numpy array that represents Exterior's rotation matrix.
        """
        ret_dmatrix_np = numpy.empty(shape=(3, 3))
        for i in range(3):
            for j in range(3):
                ret_dmatrix_np[i][j] = self._calibration[0].ext_par.dm[i][j]
        
        return ret_dmatrix_np
    
    def set_primary_point(self, cnp.ndarray prim_point_pos):
        """
        Set the camera's primary point position (a.k.a. interior orientation).
        
        Arguments:
        prim_point_pos - a 3 element array holding the values of x and y shift
            of point from sensor middle and sensor-point distance, in this 
            order.
        """
        if (<object>prim_point_pos).shape != (3,):
            raise ValueError("Expected a 3-element array")
        
        self._calibration[0].int_par.xh = prim_point_pos[0]
        self._calibration[0].int_par.yh = prim_point_pos[1]
        self._calibration[0].int_par.cc = prim_point_pos[2]
    
    def get_primary_point(self):
        """
        Returns the primary point position (a.k.a. interior orientation) as a 3
        element array holding the values of x and y shift of point from sensor
        middle and sensor-point distance, in this order.
        """
        ret = numpy.empty(3)
        ret[0] = self._calibration[0].int_par.xh
        ret[1] = self._calibration[0].int_par.yh
        ret[2] = self._calibration[0].int_par.cc
        return ret
        
    def set_radial_distortion(self, cnp.ndarray dist_coeffs):
        """
        Sets the parameters for the image radial distortion, where the x/y
        coordinates are corrected by a polynomial in r = sqrt(x**2 + y**):
        p = k1*r**2 + k2*r**4 + k3*r**6
        
        Arguments:
        dist_coeffs - length-3 array, holding k_i.
        """
        if (<object>dist_coeffs).shape != (3,):
            raise ValueError("Expected a 3-element array")
        
        self._calibration[0].added_par.k1 = dist_coeffs[0]
        self._calibration[0].added_par.k2 = dist_coeffs[1]
        self._calibration[0].added_par.k3 = dist_coeffs[2]
        
    def get_radial_distortion(self):
        """
        Returns the radial distortion polynomial coefficients as a 3 element
        array, from lowest power to highest.
        """
        ret = numpy.empty(3)
        ret[0] = self._calibration[0].added_par.k1
        ret[1] = self._calibration[0].added_par.k2
        ret[2] = self._calibration[0].added_par.k3
        return ret
    
    def set_decentering(self, cnp.ndarray decent):
        """
        Sets the parameters of decentering distortion (a.k.a. p1, p2, see [1]).
        
        Arguments:
        decent - array, holding p_i
        """
        if (<object>decent).shape != (2,):
            raise ValueError("Expected a 2-element array")
        
        self._calibration[0].added_par.p1 = decent[0]
        self._calibration[0].added_par.p2 = decent[1]
    
    def get_decentering(self):
        """
        Returns the decentering parameters [1] as a 2 element array, 
        (p_1, p_2).
        """
        ret = numpy.empty(2)
        ret[0] = self._calibration[0].added_par.p1
        ret[1] = self._calibration[0].added_par.p2
        return ret
    
    def set_affine_trans(self, affine):
        """
        Sets the affine transform parameters (x-scale, shear) applied to the
        image.
        
        Arguments:
        affine - array, holding (x-scale, shear) in order.
        """
        if (<object>affine).shape != (2,):
            raise ValueError("Expected a 2-element array")

        self._calibration[0].added_par.scx = affine[0]
        self._calibration[0].added_par.she = affine[1]
    
    def get_affine(self):
        """
        Returns the affine transform parameters [1] as a 2 element array, 
        (scx, she).
        """
        ret = numpy.empty(2)
        ret[0] = self._calibration[0].added_par.scx
        ret[1] = self._calibration[0].added_par.she
        return ret
    
    def set_glass_vec(self, cnp.ndarray gvec):
        """
        Sets the glass vector: a vector from the origin to the glass, directed
        normal to the glass.
        
        Arguments:
        gvec - a 3-element array, the glass vector.
        """
        if len(gvec) != 3:
            raise ValueError("Expected a 3-element array")
        
        self._calibration[0].glass_par.vec_x = gvec[0]
        self._calibration[0].glass_par.vec_y = gvec[1]
        self._calibration[0].glass_par.vec_z = gvec[2]
    
    def get_glass_vec(self):
        """
        Returns the glass vector, a 3-element array.
        """
        ret = numpy.empty(3)
        ret[0] = self._calibration[0].glass_par.vec_x
        ret[1] = self._calibration[0].glass_par.vec_y
        ret[2] = self._calibration[0].glass_par.vec_z
        return ret
    
    # Free memory
    def __dealloc__(self):
        free(self._calibration)

