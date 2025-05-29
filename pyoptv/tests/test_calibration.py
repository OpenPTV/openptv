import pytest
import numpy as np
from pyoptv.calibration import Calibration, Exterior, Interior, Glass, ap_52

def test_calibration_initialization():
    cal = Calibration()
    assert cal.ext_par.x0 == 0.0
    assert cal.ext_par.y0 == 0.0
    assert cal.ext_par.z0 == 0.0
    assert cal.ext_par.omega == 0.0
    assert cal.ext_par.phi == 0.0
    assert cal.ext_par.kappa == 0.0
    assert np.allclose(cal.ext_par.dm, np.zeros((3, 3)))
    assert cal.int_par.xh == 0.0
    assert cal.int_par.yh == 0.0
    assert cal.int_par.cc == 0.0
    assert cal.glass_par.vec_x == 0.0
    assert cal.glass_par.vec_y == 0.0
    assert cal.glass_par.vec_z == 0.0
    assert cal.added_par.k1 == 0.0
    assert cal.added_par.k2 == 0.0
    assert cal.added_par.k3 == 0.0
    assert cal.added_par.p1 == 0.0
    assert cal.added_par.p2 == 0.0
    assert cal.added_par.scx == 1.0
    assert cal.added_par.she == 0.0

def test_set_pos():
    cal = Calibration()
    pos = np.array([1.0, 2.0, 3.0])
    cal.set_pos(pos)
    assert cal.ext_par.x0 == 1.0
    assert cal.ext_par.y0 == 2.0
    assert cal.ext_par.z0 == 3.0

def test_set_angles():
    cal = Calibration()
    angs = np.array([0.1, 0.2, 0.3])
    cal.set_angles(angs)
    assert cal.ext_par.omega == 0.1
    assert cal.ext_par.phi == 0.2
    assert cal.ext_par.kappa == 0.3
    assert np.allclose(cal.ext_par.dm, Calibration.rotation_matrix(0.1, 0.2, 0.3))

def test_set_primary_point():
    cal = Calibration()
    prim_point = np.array([0.1, 0.2, 0.3])
    cal.set_primary_point(prim_point)
    assert cal.int_par.xh == 0.1
    assert cal.int_par.yh == 0.2
    assert cal.int_par.cc == 0.3

def test_set_radial_distortion():
    cal = Calibration()
    rad_dist = np.array([0.1, 0.2, 0.3])
    cal.set_radial_distortion(rad_dist)
    assert cal.added_par.k1 == 0.1
    assert cal.added_par.k2 == 0.2
    assert cal.added_par.k3 == 0.3

def test_set_decentering():
    cal = Calibration()
    decent = np.array([0.1, 0.2])
    cal.set_decentering(decent)
    assert cal.added_par.p1 == 0.1
    assert cal.added_par.p2 == 0.2

def test_set_affine_trans():
    cal = Calibration()
    affine = np.array([0.1, 0.2])
    cal.set_affine_trans(affine)
    assert cal.added_par.scx == 0.1
    assert cal.added_par.she == 0.2

def test_set_glass_vec():
    cal = Calibration()
    glass = np.array([0.1, 0.2, 0.3])
    cal.set_glass_vec(glass)
    assert cal.glass_par.vec_x == 0.1
    assert cal.glass_par.vec_y == 0.2
    assert cal.glass_par.vec_z == 0.3

def test_write_ori(tmp_path):
    cal = Calibration()
    cal.set_pos(np.array([1.0, 2.0, 3.0]))
    cal.set_angles(np.array([0.1, 0.2, 0.3]))
    cal.set_primary_point(np.array([0.1, 0.2, 0.3]))
    cal.set_radial_distortion(np.array([0.1, 0.2, 0.3]))
    cal.set_decentering(np.array([0.1, 0.2]))
    cal.set_affine_trans(np.array([0.1, 0.2]))
    cal.set_glass_vec(np.array([0.1, 0.2, 0.3]))

    ori_file = tmp_path / "test.ori"
    add_file = tmp_path / "test.addpar"
    cal.write_ori(ori_file, add_file)

    cal_read = Calibration.read_ori(ori_file, add_file)
    assert Calibration.compare_calib(cal, cal_read)

def test_read_ori(calibration_data_dir):
    ori_file = calibration_data_dir / "cam1.tif.ori"
    add_file = calibration_data_dir / "cam1.tif.addpar"
    cal = Calibration.read_ori(str(ori_file), str(add_file))
    assert cal.ext_par.x0 == 105.2632
    assert cal.ext_par.y0 == 102.7458
    assert cal.ext_par.z0 == 403.8822
    assert cal.ext_par.omega == -0.2383291
    assert cal.ext_par.phi == 0.2442810
    assert cal.ext_par.kappa == 0.0552577
    assert np.allclose(cal.ext_par.dm, np.array([[0.9688305, -0.0535899, 0.2418587],
                                                 [-0.0033422, 0.9734041, 0.2290704],
                                                 [-0.2477021, -0.2227387, 0.9428845]]))
    assert cal.int_par.xh == -2.4742
    assert cal.int_par.yh == 3.2567
    assert cal.int_par.cc == 100.0000
    assert cal.glass_par.vec_x == 0.0001
    assert cal.glass_par.vec_y == 0.00001
    assert cal.glass_par.vec_z == 150.0
    assert cal.added_par.k1 == 0.0
    assert cal.added_par.k2 == 0.0
    assert cal.added_par.k3 == 0.0
    assert cal.added_par.p1 == 0.0
    assert cal.added_par.p2 == 0.0
    assert cal.added_par.scx == 1.0
    assert cal.added_par.she == 0.0
