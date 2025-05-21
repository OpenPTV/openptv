import pytest
import numpy as np
from pyoptv.multimed import (
    get_mmf_from_mmlut, multimed_nlay, multimed_r_nlay, trans_Cam_Point,
    back_trans_Point, move_along_ray, init_mmlut, volumedimension
)

def test_get_mmf_from_mmlut():
    class MMLUT:
        def __init__(self):
            self.rw = 1.0
            self.origin = np.array([0.0, 0.0, 0.0])
            self.nz = 2
            self.nr = 2
            self.data = np.array([[1.0, 2.0], [3.0, 4.0]])

    class Calibration:
        def __init__(self):
            self.mmlut = MMLUT()

    cal = Calibration()
    pos = np.array([0.5, 0.5, 0.5])
    result = get_mmf_from_mmlut(cal, pos)
    assert result == 2.5

def test_multimed_nlay():
    class MM:
        def __init__(self):
            self.n1 = 1
            self.nlay = 1
            self.n2 = [1]
            self.n3 = 1

    class ExtPar:
        def __init__(self):
            self.x0 = 0.0
            self.y0 = 0.0
            self.z0 = 0.0

    class Calibration:
        def __init__(self):
            self.ext_par = ExtPar()
            self.mmlut = None

    cal = Calibration()
    mm = MM()
    pos = np.array([1.0, 1.0, 1.0])
    Xq, Yq = multimed_nlay(cal, mm, pos)
    assert Xq == 1.0
    assert Yq == 1.0

def test_multimed_r_nlay():
    class MM:
        def __init__(self):
            self.n1 = 1
            self.nlay = 1
            self.n2 = [1]
            self.n3 = 1

    class Calibration:
        def __init__(self):
            self.mmlut = None

    cal = Calibration()
    mm = MM()
    pos = np.array([1.0, 1.0, 1.0])
    result = multimed_r_nlay(cal, mm, pos)
    assert result == 1.0

def test_trans_Cam_Point():
    class ExtPar:
        def __init__(self):
            self.x0 = 0.0
            self.y0 = 0.0
            self.z0 = 0.0

    class Glass:
        def __init__(self):
            self.vec_x = 1.0
            self.vec_y = 1.0
            self.vec_z = 1.0

    class MM:
        def __init__(self):
            self.d = [1.0]

    ex = ExtPar()
    mm = MM()
    gl = Glass()
    pos = np.array([1.0, 1.0, 1.0])
    ex_t, pos_t, cross_p, cross_c = trans_Cam_Point(ex, mm, gl, pos)
    assert ex_t.x0 == 0.0
    assert ex_t.y0 == 0.0
    assert ex_t.z0 == 1.0
    assert np.allclose(pos_t, np.array([0.0, 0.0, 0.0]))

def test_back_trans_Point():
    class Glass:
        def __init__(self):
            self.vec_x = 1.0
            self.vec_y = 1.0
            self.vec_z = 1.0

    class MM:
        def __init__(self):
            self.d = [1.0]

    pos_t = np.array([1.0, 1.0, 1.0])
    mm = MM()
    G = Glass()
    cross_p = np.array([1.0, 1.0, 1.0])
    cross_c = np.array([1.0, 1.0, 1.0])
    result = back_trans_Point(pos_t, mm, G, cross_p, cross_c)
    assert np.allclose(result, np.array([0.0, 0.0, 0.0]))

def test_move_along_ray():
    glob_Z = 1.0
    vertex = np.array([0.0, 0.0, 0.0])
    direct = np.array([1.0, 1.0, 1.0])
    result = move_along_ray(glob_Z, vertex, direct)
    assert np.allclose(result, np.array([1.0, 1.0, 1.0]))

def test_init_mmlut():
    class VPar:
        def __init__(self):
            self.Zmin_lay = [0.0]
            self.Zmax_lay = [1.0]

    class CPar:
        def __init__(self):
            self.imx = 1.0
            self.imy = 1.0
            self.mm = None

    class ExtPar:
        def __init__(self):
            self.x0 = 0.0
            self.y0 = 0.0
            self.z0 = 0.0

    class IntPar:
        def __init__(self):
            self.xh = 0.0
            self.yh = 0.0

    class GlassPar:
        def __init__(self):
            self.vec_x = 1.0
            self.vec_y = 1.0
            self.vec_z = 1.0

    class AddedPar:
        def __init__(self):
            pass

    class MMLUT:
        def __init__(self):
            self.data = None

    class Calibration:
        def __init__(self):
            self.ext_par = ExtPar()
            self.int_par = IntPar()
            self.glass_par = GlassPar()
            self.added_par = AddedPar()
            self.mmlut = MMLUT()

    vpar = VPar()
    cpar = CPar()
    cal = Calibration()
    init_mmlut(vpar, cpar, cal)
    assert cal.mmlut.data is not None

def test_volumedimension():
    class VPar:
        def __init__(self):
            self.Zmin_lay = [0.0]
            self.Zmax_lay = [1.0]

    class CPar:
        def __init__(self):
            self.imx = 1.0
            self.imy = 1.0
            self.num_cams = 1
            self.mm = None

    class ExtPar:
        def __init__(self):
            self.x0 = 0.0
            self.y0 = 0.0
            self.z0 = 0.0

    class IntPar:
        def __init__(self):
            self.xh = 0.0
            self.yh = 0.0

    class GlassPar:
        def __init__(self):
            self.vec_x = 1.0
            self.vec_y = 1.0
            self.vec_z = 1.0

    class AddedPar:
        def __init__(self):
            pass

    class MMLUT:
        def __init__(self):
            self.data = None

    class Calibration:
        def __init__(self):
            self.ext_par = ExtPar()
            self.int_par = IntPar()
            self.glass_par = GlassPar()
            self.added_par = AddedPar()
            self.mmlut = MMLUT()

    vpar = VPar()
    cpar = CPar()
    cal = [Calibration()]
    xmax, xmin, ymax, ymin, zmax, zmin = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    volumedimension(xmax, xmin, ymax, ymin, zmax, zmin, vpar, cpar, cal)
    assert xmax[0] == 0.0
    assert xmin[0] == 0.0
    assert ymax[0] == 0.0
    assert ymin[0] == 0.0
    assert zmax[0] == 1.0
    assert zmin[0] == 0.0
