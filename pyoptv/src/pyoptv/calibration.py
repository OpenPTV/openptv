import numpy as np
try:
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Calibration:
    def __init__(self, pos=None, angs=None, prim_point=None, rad_dist=None,
                 decent=None, affine=None, glass=None, ext_par=None, int_par=None,
                 glass_par=None, added_par=None):
        """
        All arguments are optional arrays, default for all is zeros except 
        affine that defaults to [1, 0].
        
        Arguments can be either parameter arrays or parameter objects.
        """
        # Initialize with parameter objects if provided, otherwise create new ones
        if ext_par is not None:
            self.ext_par = ext_par
        else:
            self.ext_par = Exterior()
            
        if int_par is not None:
            self.int_par = int_par
        else:
            self.int_par = Interior()
            
        if glass_par is not None:
            self.glass_par = glass_par
        else:
            self.glass_par = Glass()
            
        if added_par is not None:
            self.added_par = added_par
        else:
            self.added_par = ap_52()
            
        self.mmlut = mmlut()

        if pos is not None:
            self.set_pos(pos)
        if angs is not None:
            self.set_angles(angs)
        if prim_point is not None:
            self.set_primary_point(prim_point)
        if rad_dist is not None:
            self.set_radial_distortion(rad_dist)
        if decent is not None:
            self.set_decentering(decent)
        if affine is not None:
            self.set_affine_trans(affine)
        if glass is not None:
            self.set_glass_vec(glass)

    def set_pos(self, pos):
        """
        Sets exterior position.
        Parameter: pos - numpy array of 3 elements for x, y, z
        """
        if len(pos) != 3:
             raise ValueError("Illegal array argument " + str(pos) + \
                " for x, y, z. Expected array/list of 3 numbers")
        self.ext_par.x0, self.ext_par.y0, self.ext_par.z0 = pos

    def set_angles(self, angs):
        """
        Sets angles (omega, phi, kappa) and recalculates Dmatrix accordingly
        Parameter: angs - array of 3 elements.
        """
        if len(angs) != 3:
            raise ValueError("Illegal array argument " + str(angs) + \
                " for omega, phi, kappa. Expected array/list of 3 numbers")
        self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa = angs
        self.ext_par.dm = self.rotation_matrix(self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa)

    def set_primary_point(self, prim_point):
        """
        Set the camera's primary point position (a.k.a. interior orientation).
        
        Arguments:
        prim_point - a 3 element array holding the values of x and y shift
            of point from sensor middle and sensor-point distance, in this 
            order.
        """
        if len(prim_point) != 3:
            raise ValueError("Expected a 3-element array")
        self.int_par.xh, self.int_par.yh, self.int_par.cc = prim_point

    def set_radial_distortion(self, rad_dist):
        """
        Sets the parameters for the image radial distortion, where the x/y
        coordinates are corrected by a polynomial in r = sqrt(x**2 + y**2):
        p = k1*r**2 + k2*r**4 + k3*r**6
        
        Arguments:
        rad_dist - length-3 array, holding k_i.
        """
        if len(rad_dist) != 3:
            raise ValueError("Expected a 3-element array")
        self.added_par.k1, self.added_par.k2, self.added_par.k3 = rad_dist

    def set_decentering(self, decent):
        """
        Sets the parameters of decentering distortion (a.k.a. p1, p2).
        
        Arguments:
        decent - array, holding p_i
        """
        if len(decent) != 2:
            raise ValueError("Expected a 2-element array")
        self.added_par.p1, self.added_par.p2 = decent

    def set_affine_trans(self, affine):
        """
        Sets the affine transform parameters (x-scale, shear) applied to the
        image.
        
        Arguments:
        affine - array, holding (x-scale, shear) in order.
        """
        if len(affine) != 2:
            raise ValueError("Expected a 2-element array")
        self.added_par.scx, self.added_par.she = affine

    def set_glass_vec(self, glass):
        """
        Sets the glass vector: a vector from the origin to the glass, directed
        normal to the glass.
        
        Arguments:
        glass - a 3-element array, the glass vector.
        """
        if len(glass) != 3:
            raise ValueError("Expected a 3-element array")
        self.glass_par.vec_x, self.glass_par.vec_y, self.glass_par.vec_z = glass

    @staticmethod
    def rotation_matrix(omega, phi, kappa):
        if HAS_NUMBA:
            return Calibration._rotation_matrix_numba(omega, phi, kappa)
        else:
            return Calibration._rotation_matrix_numpy(omega, phi, kappa)
    
    @staticmethod
    def _rotation_matrix_numpy(omega, phi, kappa):
        cp = np.cos(phi)
        sp = np.sin(phi)
        co = np.cos(omega)
        so = np.sin(omega)
        ck = np.cos(kappa)
        sk = np.sin(kappa)

        dm = np.zeros((3, 3))
        dm[0, 0] = cp * ck
        dm[0, 1] = -cp * sk
        dm[0, 2] = sp
        dm[1, 0] = co * sk + so * sp * ck
        dm[1, 1] = co * ck - so * sp * sk
        dm[1, 2] = -so * cp
        dm[2, 0] = so * sk - co * sp * ck
        dm[2, 1] = so * ck + co * sp * sk
        dm[2, 2] = co * cp

        return dm

    def write_ori(self, filename, add_file=None):
        with open(filename, 'w') as f:
            f.write(f"{self.ext_par.x0:.8f} {self.ext_par.y0:.8f} {self.ext_par.z0:.8f}\n")
            f.write(f"{self.ext_par.omega:.8f} {self.ext_par.phi:.8f} {self.ext_par.kappa:.8f}\n")
            for row in self.ext_par.dm:
                f.write(f"{row[0]:.7f} {row[1]:.7f} {row[2]:.7f}\n")
            f.write(f"{self.int_par.xh:.4f} {self.int_par.yh:.4f}\n")
            f.write(f"{self.int_par.cc:.4f}\n")
            f.write(f"{self.glass_par.vec_x:.15f} {self.glass_par.vec_y:.15f} {self.glass_par.vec_z:.15f}\n")

        if add_file:
            with open(add_file, 'w') as f:
                f.write(f"{self.added_par.k1:.8f} {self.added_par.k2:.8f} {self.added_par.k3:.8f} ")
                f.write(f"{self.added_par.p1:.8f} {self.added_par.p2:.8f} ")
                f.write(f"{self.added_par.scx:.8f} {self.added_par.she:.8f}")

    @staticmethod
    def read_ori(filename, add_file=None, add_fallback=None):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]

        cal = Calibration()
        
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0 = map(float, lines[0].split())
        cal.ext_par.omega, cal.ext_par.phi, cal.ext_par.kappa = map(float, lines[1].split())
        cal.ext_par.dm = np.array([list(map(float, line.split())) for line in lines[2:5]])
        cal.int_par.xh, cal.int_par.yh = map(float, lines[5].split())
        cal.int_par.cc = float(lines[6])
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z = map(float, lines[7].split())

        if add_file or add_fallback:
            try:
                with open(add_file, 'r') as f:
                    add_lines = f.readlines()
            except (FileNotFoundError, TypeError):
                if add_fallback:
                    with open(add_fallback, 'r') as f:
                        add_lines = f.readlines()
                else:
                    # If no add file, use defaults
                    return cal

            cal.added_par.k1, cal.added_par.k2, cal.added_par.k3, cal.added_par.p1, cal.added_par.p2, cal.added_par.scx, cal.added_par.she = map(float, add_lines[0].split())

        return cal

    @staticmethod
    def compare_calib(c1, c2):
        return (Calibration.compare_exterior(c1.ext_par, c2.ext_par) and
                Calibration.compare_interior(c1.int_par, c2.int_par) and
                Calibration.compare_glass(c1.glass_par, c2.glass_par) and
                Calibration.compare_addpar(c1.added_par, c2.added_par))

    @staticmethod
    def compare_exterior(e1, e2):
        return (np.allclose(e1.dm, e2.dm) and
                e1.x0 == e2.x0 and e1.y0 == e2.y0 and e1.z0 == e2.z0 and
                e1.omega == e2.omega and e1.phi == e2.phi and e1.kappa == e2.kappa)

    @staticmethod
    def compare_interior(i1, i2):
        return (i1.xh == i2.xh and i1.yh == i2.yh and i1.cc == i2.cc)

    @staticmethod
    def compare_glass(g1, g2):
        return (g1.vec_x == g2.vec_x and g1.vec_y == g2.vec_y and g1.vec_z == g2.vec_z)

    @staticmethod
    def compare_addpar(a1, a2):
        return (a1.k1 == a2.k1 and a1.k2 == a2.k2 and a1.k3 == a2.k3 and
                a1.p1 == a2.p1 and a1.p2 == a2.p2 and a1.scx == a2.scx and a1.she == a2.she)

    @staticmethod
    def read_calibration(ori_file, add_file=None, fallback_file=None):
        return Calibration.read_ori(ori_file, add_file, fallback_file)

    def write_calibration(self, filename, add_file=None):
        self.write_ori(filename, add_file)

    def get_pos(self):
        """
        Returns numpy array of 3 elements representing exterior's x, y, z
        """
        return np.array([self.ext_par.x0, self.ext_par.y0, self.ext_par.z0])

    def get_angles(self):
        """
        Returns a numpy array of 3 elements representing omega, phi, kappa
        """
        return np.array([self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa])

    def get_rotation_matrix(self):
        """
        Returns a 3x3 numpy array that represents Exterior's rotation matrix.
        """
        return self.ext_par.dm.copy()

    def get_primary_point(self):
        """
        Returns the primary point position (a.k.a. interior orientation) as a 3
        element array holding the values of x and y shift of point from sensor
        middle and sensor-point distance, in this order.
        """
        return np.array([self.int_par.xh, self.int_par.yh, self.int_par.cc])

    def get_radial_distortion(self):
        """
        Returns the radial distortion polynomial coefficients as a 3 element
        array, from lowest power to highest.
        """
        return np.array([self.added_par.k1, self.added_par.k2, self.added_par.k3])

    def get_decentering(self):
        """
        Returns the decentering parameters as a 2 element array, (p_1, p_2).
        """
        return np.array([self.added_par.p1, self.added_par.p2])

    def get_affine(self):
        """
        Returns the affine transform parameters as a 2 element array, (scx, she).
        """
        return np.array([self.added_par.scx, self.added_par.she])

    def get_glass_vec(self):
        """
        Returns the glass vector, a 3-element array.
        """
        return np.array([self.glass_par.vec_x, self.glass_par.vec_y, self.glass_par.vec_z])

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
        # Convert bytes to string if needed
        if isinstance(ori_file, bytes):
            ori_file = ori_file.decode('utf-8')
        if isinstance(add_file, bytes):
            add_file = add_file.decode('utf-8')
        if isinstance(fallback_file, bytes):
            fallback_file = fallback_file.decode('utf-8')
            
        cal = self.read_ori(ori_file, add_file, fallback_file)
        self.ext_par = cal.ext_par
        self.int_par = cal.int_par
        self.glass_par = cal.glass_par
        self.added_par = cal.added_par

    def write(self, filename, add_file):
        """
        Write the calibration data to disk. Uses two output file, one for the
        linear calibration part, and one for distortion parameters.
        
        Arguments:
        filename - path to file containing exterior, interior and glass
            parameters.
        add_file - optional path to file containing distortion parameters.
        """
        # Convert bytes to string if needed
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        if isinstance(add_file, bytes):
            add_file = add_file.decode('utf-8')
            
        self.write_ori(filename, add_file)

class Exterior:
    def __init__(self):
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.omega = 0.0
        self.phi = 0.0
        self.kappa = 0.0
        self.dm = np.zeros((3, 3))

class Interior:
    def __init__(self):
        self.xh = 0.0
        self.yh = 0.0
        self.cc = 0.0

class Glass:
    def __init__(self):
        self.vec_x = 0.0
        self.vec_y = 0.0
        self.vec_z = 0.0

class ap_52:
    def __init__(self):
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.scx = 1.0
        self.she = 0.0

class mmlut:
    def __init__(self):
        self.origin = np.zeros(3)
        self.nr = 0
        self.nz = 0
        self.rw = 0
        self.data = None

# Add numba-compiled version if numba is available
if HAS_NUMBA:
    
    def _rotation_matrix_numba(omega, phi, kappa):
        cp = np.cos(phi)
        sp = np.sin(phi)
        co = np.cos(omega)
        so = np.sin(omega)
        ck = np.cos(kappa)
        sk = np.sin(kappa)

        dm = np.zeros((3, 3))
        dm[0, 0] = cp * ck
        dm[0, 1] = -cp * sk
        dm[0, 2] = sp
        dm[1, 0] = co * sk + so * sp * ck
        dm[1, 1] = co * ck - so * sp * sk
        dm[1, 2] = -so * cp
        dm[2, 0] = so * sk - co * sp * ck
        dm[2, 1] = so * ck + co * sp * sk
        dm[2, 2] = co * cp

        return dm
    
    # Assign the numba version to the Calibration class
    Calibration._rotation_matrix_numba = staticmethod(_rotation_matrix_numba)
