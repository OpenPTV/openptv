class Dmatrix:
    def __init__(self):
        self.values = [[0.0] * 3 for _ in range(3)]
        
class Exterior:
    def __init__(self):
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.omega = 0.0
        self.phi = 0.0
        self.kappa = 0.0
        self.dm = Dmatrix()
        
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
        self.scx = 0.0
        self.she = 0.0
        
class vec3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
class mmlut:
    def __init__(self):
        self.origin = vec3d(0.0, 0.0, 0.0)
        self.nr = 0
        self.nz = 0
        self.rw = 0
        self.data = None
        
class Calibration:
    def __init__(self):
        self.ext_par = Exterior()
        self.int_par = Interior()
        self.glass_par = Glass()
        self.added_par = ap_52()
        self.mmlut = mmlut()
