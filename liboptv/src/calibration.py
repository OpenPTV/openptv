
Dmatrix = [[0.0 for i in range(3)] for j in range(3)]

class Calibration:
    class Exterior:
        def __init__(self):
            self.dm = [[0 for j in range(3)] for i in range(3)]
            self.omega = 0.0
            self.phi = 0.0
            self.kappa = 0.0
            self.x0 = 0.0
            self.y0 = 0.0
            self.z0 = 0.0

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

    class mmlut:
        def __init__(self):
            self.origin = vec3d()
            self.nr = 0
            self.nz = 0
            self.rw = 0
            self.data = []

    def __init__(self):
        self.ext_par = Calibration.Exterior()
        self.int_par = Calibration.Interior()
        self.glass_par = Calibration.Glass()
        self.added_par = Calibration.ap_52()
        self.mmlut = Calibration.mmlut()


def write_ori(Ex, I, G, ap, filename, add_file):
    """Write exterior and interior orientation, and - if available, parameters for
    distortion corrections.
    
    Arguments:
    Exterior Ex - exterior orientation.
    Interior I - interior orientation.
    Glass G - glass parameters.
    ap_52 addp - optional additional (distortion) parameters. NULL is fine if
       add_file is NULL.
    char *filename - path of file to contain interior, exterior and glass
       orientation data.
    char *add_file - path of file to contain added (distortions) parameters.
    """
    success = 0
    try:
        with open(filename, 'w') as fp:
            fp.write("{:11.8f} {:11.8f} {:11.8f}\n    {:10.8f}  {:10.8f}  {:10.8f}\n\n".format(
                Ex.x0, Ex.y0, Ex.z0, Ex.omega, Ex.phi, Ex.kappa))
            for i in range(3):
                fp.write("    {:10.7f} {:10.7f} {:10.7f}\n".format(
                    Ex.dm[i][0], Ex.dm[i][1], Ex.dm[i][2]))
            fp.write("\n    {:8.4f} {:8.4f}\n    {:8.4f}\n".format(I.xh, I.yh, I.cc))
            fp.write("\n    {:20.15f} {:20.15f}  {:20.15f}\n".format(G.vec_x, G.vec_y, G.vec_z))
    except IOError:
        print("Can't open ascii file: {}".format(filename))
        return success
    
    if add_file is None:
        return success
    
    try:
        with open(add_file, 'w') as fp:
            fp.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}".format(
                ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she))
            success = 1
    except IOError:
        print("Can't open ascii file: {}".format(add_file))
        return success
    
    return success


def read_ori (Ex, I, G, ori_file, addp, add_file, add_fallback):
    """
    Reads the orientation file and the additional parameters file.
    """
    fp = open (ori_file, "r")
    if not fp:
        print("Can't open ORI file: %s\n", ori_file)
        return 0
    
    # Exterior
    scan_res = fscanf (fp, "%lf %lf %lf %lf %lf %lf",
	  &(Ex->x0), &(Ex->y0), &(Ex->z0),
	  &(Ex->omega), &(Ex->phi), &(Ex->kappa))
    if (scan_res != 6):
        return 0
    
    # Exterior rotation matrix
    for i in range(3):
        scan_res = fscanf (fp, " %lf %lf %lf",
            &(Ex->dm[i][0]), &(Ex->dm[i][1]), &(Ex->dm[i][2]))
        if (scan_res != 3):
            return 0
    
    # Interior
    scan_res = fscanf (fp, "%lf %lf %lf", &(I->xh), &(I->yh), &(I->cc))
    if (scan_res != 3):
        return 0
    
    # Glass
    scan_res = fscanf (fp, "%lf %lf %lf", &(G->vec_x), &(G->vec_y), &(G->vec_z))
    if (scan_res != 3):
        return 0
    
    fp.close()
    
    # Additional:
    fp = open(add_file, "r")
    if ((fp == NULL) and add_fallback):
        fp = open (add_fallback, "r")
    
    if fp:
        scan_res = fscanf (fp, "%lf %lf %lf %lf %lf %lf %lf",
            &(addp->k1), &(addp->k2), &(addp->k3), &(addp->p1), &(addp->p2),
            &(addp->scx), &(addp->she))
        fp.close()
    else:
        print("no addpar fallback used\n") # Waits for proper logging.
        addp->k1 = addp->k2 = addp->k3 = addp->p1 = addp->p2 = addp->she = 0.0
        addp->scx=1.0
    
    return 1

def compare_exterior(e1, e2):
    for row in range(3):
        for col in range(3):
            if e1.dm[row][col] != e2.dm[row][col]:
                return 0
    return ((e1.x0 == e2.x0) and (e1.y0 == e2.y0) and (e1.z0 == e2.z0)\
        and (e1.omega == e2.omega) and (e1.phi == e2.phi) \
        and (e1.kappa == e2.kappa))
    
    
def compare_interior(i1, i2):
    return i1.xh == i2.xh and i1.yh == i2.yh and i1.cc == i2.cc


def compare_glass(g1, g2):
    """
    This function takes two arguments `g1` and `g2`, which are `Glass` objects that need to be compared. The function then returns `1` if all `vec_x`, `vec_y` and `vec_z` values of `g1` are equal to the corresponding values in `g2`. Else, the function returns `0`.

    Args:
        g1 (_type_): _description_
        g2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return g1.vec_x == g2.vec_x and g1.vec_y == g2.vec_y and g1.vec_z == g2.vec_z



import unittest

class ap_52:
    def __init__(self, k1, k2, k3, p1, p2, scx, she):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.scx = scx
        self.she = she

def compare_addpar(a1, a2):
    return (a1.k1 == a2.k1) and (a1.k2 == a2.k2) and (a1.k3 == a2.k3) and \
        (a1.p1 == a2.p1) and (a1.p2 == a2.p2) and (a1.scx == a2.scx) and \
        (a1.she == a2.she)

class TestCompareAddpar(unittest.TestCase):
    def test_compare_addpar(self):
        a1 = ap_52(1, 2, 3, 4, 5, 6, 7)
        a2 = ap_52(1, 2, 3, 4, 5, 6, 7)
        self.assertTrue(compare_addpar(a1, a2))
        
        a3 = ap_52(1, 2, 3, 4, 6, 6, 7)
        self.assertFalse(compare_addpar(a1, a3))

def read_calibration(ori_file, add_file, fallback_file):
    ret = Calibration()
    
    # indicate that data is not set yet
    ret.mmlut.data = None
    
    if read_ori(ret.ext_par, ret.int_par, ret.glass_par, ori_file, ret.added_par,
                add_file, fallback_file):
        rotation_matrix(ret.ext_par)
        return ret
    else:
        free(ret)
        return None
    

def write_calibration(cal, ori_file, add_file):
    return write_ori(cal.ext_par, cal.int_par, cal.glass_par, cal.added_par, ori_file, add_file)


def rotation_matrix(Ex):
    
    import math
    
    # Calculate the necessary trigonometric functions to rotate the Dmatrix of Exterior Ex
    cp = math.cos(Ex.phi)
    sp = math.sin(Ex.phi)
    co = math.cos(Ex.omega)
    so = math.sin(Ex.omega)
    ck = math.cos(Ex.kappa)
    sk = math.sin(Ex.kappa)
    
    # Modify the Exterior Ex with the new Dmatrix
    Ex.dm[0][0] = cp * ck
    Ex.dm[0][1] = -cp * sk
    Ex.dm[0][2] = sp
    Ex.dm[1][0] = co * sk + so * sp * ck
    Ex.dm[1][1] = co * ck - so * sp * sk
    Ex.dm[1][2] = -so * cp
    Ex.dm[2][0] = so * sk - co * sp * ck
    Ex.dm[2][1] = so * ck + co * sp * sk
    Ex.dm[2][2] = co * cp