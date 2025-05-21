import numpy as np
import numba
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Calibration:
    def __init__(self, pos=None, angs=None, prim_point=None, rad_dist=None,
                 decent=None, affine=None, glass=None):
        self.ext_par = Exterior()
        self.int_par = Interior()
        self.glass_par = Glass()
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
        self.ext_par.x0, self.ext_par.y0, self.ext_par.z0 = pos

    def set_angles(self, angs):
        self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa = angs
        self.ext_par.dm = self.rotation_matrix(self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa)

    def set_primary_point(self, prim_point):
        self.int_par.xh, self.int_par.yh, self.int_par.cc = prim_point

    def set_radial_distortion(self, rad_dist):
        self.added_par.k1, self.added_par.k2, self.added_par.k3 = rad_dist

    def set_decentering(self, decent):
        self.added_par.p1, self.added_par.p2 = decent

    def set_affine_trans(self, affine):
        self.added_par.scx, self.added_par.she = affine

    def set_glass_vec(self, glass):
        self.glass_par.vec_x, self.glass_par.vec_y, self.glass_par.vec_z = glass

    @staticmethod
    @numba.jit(nopython=True)
    def rotation_matrix(omega, phi, kappa):
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

        ext_par = Exterior()
        int_par = Interior()
        glass_par = Glass()
        added_par = ap_52()

        ext_par.x0, ext_par.y0, ext_par.z0 = map(float, lines[0].split())
        ext_par.omega, ext_par.phi, ext_par.kappa = map(float, lines[1].split())
        ext_par.dm = np.array([list(map(float, line.split())) for line in lines[2:5]])
        int_par.xh, int_par.yh = map(float, lines[5].split())
        int_par.cc = float(lines[6])
        glass_par.vec_x, glass_par.vec_y, glass_par.vec_z = map(float, lines[7].split())

        if add_file or add_fallback:
            try:
                with open(add_file, 'r') as f:
                    add_lines = f.readlines()
            except FileNotFoundError:
                with open(add_fallback, 'r') as f:
                    add_lines = f.readlines()

            added_par.k1, added_par.k2, added_par.k3, added_par.p1, added_par.p2, added_par.scx, added_par.she = map(float, add_lines[0].split())

        return Calibration(ext_par, int_par, glass_par, added_par)

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
