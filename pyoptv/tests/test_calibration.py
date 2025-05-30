from pathlib import Path
import pytest
import numpy as np
from pyoptv.calibration import Calibration, Exterior, Interior, Glass, ap_52
from pyoptv.calibration import read_ori

# Helper to create a calibration object with example values
# matching those in the files read by test_read_ori

def test_calibration_rotation_angles():
    # Test rotation matrices for omega, phi, kappa = pi/2
    from math import pi
    ex = Exterior()
    # omega
    ex.omega = pi/2
    ex.phi = 0
    ex.kappa = 0
    ex.x0 = ex.y0 = ex.z0 = 0
    ex.dm = np.zeros((3,3))
    ex.update_rotation_matrix()
    rotx = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    assert np.allclose(ex.dm, rotx, atol=1e-6)

    # phi
    ex = Exterior()
    ex.omega = 0
    ex.phi = pi/2
    ex.kappa = 0
    ex.x0 = ex.y0 = ex.z0 = 0
    ex.dm = np.zeros((3,3))
    ex.update_rotation_matrix()
    roty = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    assert np.allclose(ex.dm, roty, atol=1e-6)

    # kappa
    ex = Exterior()
    ex.omega = 0
    ex.phi = 0
    ex.kappa = pi/2
    ex.x0 = ex.y0 = ex.z0 = 0
    ex.dm = np.zeros((3,3))
    ex.update_rotation_matrix()
    rotz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert np.allclose(ex.dm, rotz, atol=1e-6)

@pytest.fixture(scope="session")
def testing_fodder_dir():
    """Fixture to provide the path to the testing_fodder directory inside pyoptv/tests."""
    return Path(__file__).parent / "testing_fodder"

# Regression test for reading orientation files
def test_read_ori(testing_fodder_dir):
    ori_file = testing_fodder_dir / "calibration" / "cam1.tif.ori"
    add_file = testing_fodder_dir / "calibration" / "cam1.tif.addpar"
    cal = read_ori(str(ori_file), str(add_file))
    # Compare to known correct values
    assert np.isclose(cal.ext_par.x0, 105.2632, atol=1e-4)
    assert np.isclose(cal.ext_par.y0, 102.7458, atol=1e-4)
    assert np.isclose(cal.ext_par.z0, 403.8822, atol=1e-4)
    assert np.isclose(cal.ext_par.omega, -0.2383291, atol=1e-6)
    assert np.isclose(cal.ext_par.phi, 0.2442810, atol=1e-6)
    assert np.isclose(cal.ext_par.kappa, 0.0552577, atol=1e-6)
    assert np.allclose(cal.ext_par.dm, np.array([[0.9688305, -0.0535899, 0.2418587],
                                                 [-0.0033422, 0.9734041, 0.2290704],
                                                 [-0.2477021, -0.2227387, 0.9428845]]), atol=1e-6)
    assert np.isclose(cal.int_par.xh, -2.4742, atol=1e-4)
    assert np.isclose(cal.int_par.yh, 3.2567, atol=1e-4)
    assert np.isclose(cal.int_par.cc, 100.0000, atol=1e-4)
    assert np.isclose(cal.glass_par.vec_x, 0.0001, atol=1e-6)
    assert np.isclose(cal.glass_par.vec_y, 0.00001, atol=1e-7)
    assert np.isclose(cal.glass_par.vec_z, 150.0, atol=1e-4)
    assert np.isclose(cal.added_par.k1, 0.0, atol=1e-6)
    assert np.isclose(cal.added_par.k2, 0.0, atol=1e-6)
    assert np.isclose(cal.added_par.k3, 0.0, atol=1e-6)
    assert np.isclose(cal.added_par.p1, 0.0, atol=1e-6)
    assert np.isclose(cal.added_par.p2, 0.0, atol=1e-6)
    assert np.isclose(cal.added_par.scx, 1.0, atol=1e-6)
    assert np.isclose(cal.added_par.she, 0.0, atol=1e-6)

# Unit test for writing orientation files
def test_write_ori(tmp_path: Path):
    # Create a calibration object with known values
    ext = Exterior(105.2632, 102.7458, 403.8822, -0.2383291, 0.2442810, 0.0552577)
    ext.update_rotation_matrix()
    intp = Interior(-2.4742, 3.2567, 100.0)
    glass = Glass(0.0001, 0.00001, 150.0)
    addp = ap_52(0., 0., 0., 0., 0., 1., 0.)
    cal = Calibration(ext_par = ext, int_par = intp, glass_par = glass, added_par = addp)
    ori_file = tmp_path / "test.ori"
    add_file = tmp_path / "test.addpar"
    cal.write_ori(ori_file, add_file)
    cal_read = read_ori(str(ori_file), str(add_file))
    # Use a compare_calib method if available, else compare fields
    assert np.isclose(cal.ext_par.x0, cal_read.ext_par.x0, atol=1e-6)
    assert np.isclose(cal.ext_par.y0, cal_read.ext_par.y0, atol=1e-6)
    assert np.isclose(cal.ext_par.z0, cal_read.ext_par.z0, atol=1e-6)
    assert np.isclose(cal.ext_par.omega, cal_read.ext_par.omega, atol=1e-6)
    assert np.isclose(cal.ext_par.phi, cal_read.ext_par.phi, atol=1e-6)
    assert np.isclose(cal.ext_par.kappa, cal_read.ext_par.kappa, atol=1e-6)
    assert np.allclose(cal.ext_par.dm, cal_read.ext_par.dm, atol=1e-6)
    assert np.isclose(cal.int_par.xh, cal_read.int_par.xh, atol=1e-6)
    assert np.isclose(cal.int_par.yh, cal_read.int_par.yh, atol=1e-6)
    assert np.isclose(cal.int_par.cc, cal_read.int_par.cc, atol=1e-6)
    assert np.isclose(cal.glass_par.vec_x, cal_read.glass_par.vec_x, atol=1e-6)
    assert np.isclose(cal.glass_par.vec_y, cal_read.glass_par.vec_y, atol=1e-6)
    assert np.isclose(cal.glass_par.vec_z, cal_read.glass_par.vec_z, atol=1e-6)
    assert np.isclose(cal.added_par.k1, cal_read.added_par.k1, atol=1e-6)
    assert np.isclose(cal.added_par.k2, cal_read.added_par.k2, atol=1e-6)
    assert np.isclose(cal.added_par.k3, cal_read.added_par.k3, atol=1e-6)
    assert np.isclose(cal.added_par.p1, cal_read.added_par.p1, atol=1e-6)
    assert np.isclose(cal.added_par.p2, cal_read.added_par.p2, atol=1e-6)
    assert np.isclose(cal.added_par.scx, cal_read.added_par.scx, atol=1e-6)
    assert np.isclose(cal.added_par.she, cal_read.added_par.she, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])