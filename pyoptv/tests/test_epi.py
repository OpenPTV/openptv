import pytest
import numpy as np
import os
from pyoptv.epi import epipolar_curve
from pyoptv.calibration import Calibration
from pyoptv.parameters import ControlPar, VolumePar, read_control_par, read_volume_par


def test_epipolar_curve_two_cameras(calibration_data_dir, test_data_dir):
    ori_tmpl = os.path.join(calibration_data_dir, "sym_cam{cam_num}.tif.ori")
    add_file = os.path.join(calibration_data_dir, "cam1.tif.addpar")

    orig_cal = Calibration()
    orig_cal.read_ori(ori_tmpl.format(cam_num=1), add_file)
    proj_cal = Calibration()
    proj_cal.read_ori(ori_tmpl.format(cam_num=3), add_file)

    # reorient cams:
    orig_cal.set_angles([0., -np.pi/4., 0.])
    proj_cal.set_angles([0., 3*np.pi/4., 0.])

    cpar = ControlPar(4)  # or use the correct number of cameras if known
    cpar = read_control_par(str(test_data_dir / "corresp" / "control.par"))
    sens_size = [cpar.imx, cpar.imy]

    vpar = read_volume_par(str(test_data_dir / "corresp" / "criteria.par"))
    vpar.Zmin_lay = [-10, -10]
    vpar.Zmax_lay = [10, 10]

    # Central point translates to central point because cameras point directly at each other.
    mid = np.array(sens_size) / 2.
    line = epipolar_curve(mid, orig_cal, proj_cal, 5, cpar, vpar)
    assert np.all(np.abs(line - mid) < 1e-6)

    # An equatorial point draws a latitude.
    line = epipolar_curve(mid - np.array([100., 0.]), orig_cal, proj_cal, 5, cpar, vpar)
    assert np.all(np.argsort(line[:, 0]) == np.arange(5)[::-1])
    assert np.all(np.abs(line[:, 1] - mid[1]) < 1e-6)
