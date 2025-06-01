import pytest
import numpy as np
from pyoptv.epi import epi_mm, epi_mm_2D, epipolar_curve, find_candidate
from pyoptv.calibration import Calibration, Exterior, Interior, Glass, ap_52
from pyoptv.trafo import metric_to_pixel
from pyoptv.parameters import MMNP, ControlPar, VolumePar
from pyoptv.tracking_frame_buf import Target
from pyoptv.epi import Coord2D

def test_epi_mm_2D():

    test_cal = Calibration(
        Exterior(0.0, 0.0, 100.0, 0.0, 0.0, 0.0),
        Interior(0.0, 0.0, 100.0),
        Glass(0.0, 0.0, 50.0),
        ap_52(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    )
    
    test_mm = MMNP()
    test_mm.nlay = 1    
    test_mm.n1 = 1.0
    test_mm.n2 = [1.49, 0.0, 0.0]
    test_mm.d = [5.0, 0.0, 0.0]
    test_mm.n3 = 1.33
    
    test_vpar = VolumePar()
    test_vpar.X_lay = [-250.0, 250.0]
    test_vpar.Zmin_lay = [-100.0, -100.0]
    test_vpar.Zmax_lay = [100.0, 100.0]
    test_vpar.cnx = 0.3
    test_vpar.cny = 0.3
    test_vpar.cn = 0.01
    test_vpar.csumg = 0.01
    test_vpar.corrmin = 33
    test_vpar.eps0 = 1.0


    x, y = 1.0, 10.0

    out = epi_mm_2D(x, y, test_cal, test_mm, test_vpar)

    assert np.allclose(out, [0.85858163, 8.58581626, 0.0], atol=1e-5), f"Expected [0.85858163, 8.58581626, 0.0], but got {out}"

    x, y = 0.0, 0.0
    out = epi_mm_2D(x, y, test_cal, test_mm, test_vpar)

    assert np.allclose(out, [0.0, 0.0, 0.0], atol=1e-5), f"Expected [0.0, 0.0, 0.0], but got {out}"

def test_epi_mm():
    test_cal_1 = Calibration(
        Exterior(10.0, 0.0, 100.0, 0.0, -0.01, 0.0),
        Interior(0.0, 0.0, 100.0),
        Glass(0.0, 0.0, 50.0),
        ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )


    test_cal_2 = Calibration(
        Exterior(-10.0, 0.0, 100.0, 0.0, 0.01, 0.0),
        Interior(0.0, 0.0, 100.0),
        Glass(0.0, 0.0, 50.0),
        ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )

    test_mm = MMNP()
    test_mm.nlay = 1    
    test_mm.n1 = 1.0
    test_mm.n2 = [1.49, 0.0, 0.0]
    test_mm.d = [5.0, 0.0, 0.0]
    test_mm.n3 = 1.33

    test_vpar = VolumePar()
    test_vpar.X_lay = [-250.0, 250.0]
    test_vpar.Zmin_lay = [-100.0, -100.0]
    test_vpar.Zmax_lay = [100.0, 100.0]
    test_vpar.cnx = 0.3
    test_vpar.cny = 0.3
    test_vpar.cn = 0.01
    test_vpar.csumg = 0.01
    test_vpar.corrmin = 33
    test_vpar.eps0 = 1.0

    x, y = 10.0, 10.0
    xmin, xmax, ymin, ymax = epi_mm(x, y, test_cal_1, test_cal_2, test_mm, test_vpar)

    assert np.allclose([xmin, xmax, ymin, ymax], [26.44927852, 10.08218486, 51.60078764, 10.04378909], atol=1e-5), \
        f"Expected [26.44927852, 10.08218486, 51.60078764, 10.04378909], but got {[xmin, xmax, ymin, ymax]}"
    

def test_find_candidate():
    # Set of particles to choose from
    test_pix = [
        Target(pnr=0, x=0.0, y=-0.2, sumg=5, nx=1, ny=2, n=10, tnr=-999),
        Target(pnr=6, x=0.2, y=0.0, sumg=10, nx=8, ny=1, n=20, tnr=-999),
        Target(pnr=3, x=0.2, y=0.8, sumg=10, nx=3, ny=3, n=30, tnr=-999),
        Target(pnr=4, x=0.4, y=-1.1, sumg=10, nx=3, ny=3, n=40, tnr=-999),
        Target(pnr=1, x=0.7, y=-0.1, sumg=10, nx=3, ny=3, n=50, tnr=-999),
        Target(pnr=2, x=1.2, y=0.3, sumg=10, nx=3, ny=3, n=60, tnr=-999),
        Target(pnr=5, x=10.4, y=0.1, sumg=10, nx=3, ny=3, n=70, tnr=-999)
    ]

    num_pix = 7  # length of the test_pix

    # Coordinates of particles
    test_crd = [
        Coord2D(pnr=6, x=0.1, y=0.1),
        Coord2D(pnr=3, x=0.2, y=0.8),
        Coord2D(pnr=4, x=0.4, y=-1.1),
        Coord2D(pnr=1, x=0.7, y=-0.1),
        Coord2D(pnr=2, x=1.2, y=0.3),
        Coord2D(pnr=0, x=0.0, y=0.0),
        Coord2D(pnr=5, x=10.4, y=0.1)
    ]

    # Calibration parameters
    test_cal = Calibration(
        Exterior(0.0, 0.0, 100.0, 0.0, 0.0, 0.0),
        Interior(0.0, 0.0, 100.0),
        Glass(0.0, 0.0, 50.0),
        ap_52(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    )


    # Medium parameters
    test_mm = MMNP(
        nlay=1,
        n1=1.0,
        n2=[1.49, 0.0, 0.0],
        d=[5.0, 0.0, 0.0],
        n3=1.33
    )

    test_vpar = VolumePar(
        X_lay=[-250.0, 250.0],
        Zmin_lay=[-100.0, -100.0],
        Zmax_lay=[100.0, 100.0],
        cnx=0.3,
        cny=0.3,
        cn=0.01,
        csumg=0.01,
        corrmin=33,
        eps0=1.0
    )



    # Control parameters
    test_cpar = ControlPar(4)

    test_cpar.hp_flag=1
    test_cpar.allCam_flag=0
    test_cpar.tiff_flag=1
    test_cpar.imx=1280
    test_cpar.imy=1024
    test_cpar.pix_x=0.02
    test_cpar.pix_y=0.02
    test_cpar.chfield=0
    test_cpar.mm=test_mm
    

    # Epipolar line
    xa, ya, xb, yb = -10.0, -10.0, 10.0, 10.0

    # Find candidates
    candidates = find_candidate(test_crd, test_pix, xa, ya, xb, yb, test_vpar, test_cpar, test_cal)

    # Assertions
    assert candidates[0].pnr == 0, f"Expected candidate with pnr=0, but got {candidates[0]['pnr']}"
    assert candidates[0].tol < 1e-5, f"Expected tolerance < 1e-5, but got {candidates[0]['tol']}"

    sum_corr = sum(cand["corr"] for cand in candidates)
    assert abs(sum_corr - 3301.0) < 1e-5, f"Expected sum_corr=3301.0, but got {sum_corr}"
    assert len(candidates) == 5, f"Expected 5 candidates, but got {len(candidates)}"
    assert abs(candidates[3]["tol"] - 0.636396) < 1e-5, f"Expected tol=0.636396 for candidate 3, but got {candidates[3]['tol']}"

def test_epi_mm_perpendicular():
    # First camera
    test_cal_1 = Calibration(
        Exterior(0.0, 0.0, 100.0, 0.0, 0.0, 0.0),
        Interior(0.0, 0.0, 100.0),
        Glass(0.0, 0.0, 50.0),
        ap_52(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    )

    # Second camera at small angle around y-axis
    test_cal_2 = Calibration(
        Exterior(100.0, 0.0, 100.0, 0.0, 1.57, 0.0),  # 90 degrees around y-axis
        Interior(0.0, 0.0, 100.0),
        Glass(0.0, 0.0, 50.0),
        ap_52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    
    # Medium parameters
    test_mm = MMNP(
        nlay=1,
        n1=1.0,
        n2=[1.0, 0.0, 0.0],
        d=[1.0, 0.0, 0.0],
        n3=1.0
    )
    
    # Volume parameters
    test_vpar = VolumePar(
        X_lay=[-250.0, 250.0],
        Zmin_lay=[-100.0, -100.0],
        Zmax_lay=[100.0, 100.0],
        cnx=0.3,
        cny=0.3,
        cn=0.01,
        csumg=0.01,
        corrmin=33,
        eps0=1.0
    )


    # Compute epipolar line
    xmin, xmax, ymin, ymax = epipolar_curve(0.0, 0.0, test_cal_1, test_cal_2, test_mm, test_vpar)

    # Assertions
    assert np.allclose([xmin, xmax, ymin, ymax], [-100.0, 0.0, 100.0, 0.0], atol=1e-5), \
        f"Expected [-100.0, 0.0, 100.0, 0.0], but got {[xmin, xmax, ymin, ymax]}"


if __name__ == "__main__":
    pytest.main([__file__])