import pytest
from pyoptv.epi import find_candidate, Coord2D
from pyoptv.parameters import ControlPar, VolumePar, MMNP
from pyoptv.calibration import Calibration
from pyoptv.trafo import pixel_to_metric
def make_dummy_list(num, xmin, xmax, ymin, ymax):
    # generate a sorted list of Coord2D points along x with y between bounds
    pts = []
    step = (xmax - xmin) / (num - 1)
    for i in range(num):
        x = xmin + i * step
        y = (ymin + ymax) / 2
        pt = Coord2D(pnr=i, x=x, y=y)
        pts.append(pt)
    return pts

@pytest.fixture
def dummy_params():
    # minimal ControlPar and VolumePar to call find_candidate
    cpar = ControlPar()
    cpar.pix_x = 1.0; cpar.pix_y = 1.0; cpar.imx = 100; cpar.imy = 100
    vpar = VolumePar(); vpar.eps0 = 0.1; vpar.cn = vpar.cnx = vpar.cny = 0; vpar.csumg = -1
    calib = Calibration()
    calib.int_par.xh = calib.int_par.yh = 0
    return cpar, vpar, calib


def test_out_of_bounds_returns_negative(dummy_params):
    cpar, vpar, calib = dummy_params
    coords = make_dummy_list(10, 0, 1, 0, 1)
    pix = [None] * 10  # pix list unused if out-of-bounds
    # set xa/xb so epipolar line lies entirely left of sensor
    xa, ya, xb, yb = -200, -200, -150, -150
    count, cands = find_candidate(coords, pix, len(coords), xa, ya, xb, yb, 0,0,0,0, vpar, cpar, calib)
    assert count == -1
    assert cands == []

@pytest.mark.parametrize("xa,xb,expected", [
    (0, 10, 10), (5, 15, 6), (10, 20, 0)
])
def test_in_bounds_candidate_counts(dummy_params, xa, xb, expected):
    cpar, vpar, calib = dummy_params
    ymin, ymax = 0, 0
    coords = make_dummy_list(20, 0, 20, ymin, ymax)
    # simple horizontal epipolar: y=0
    count, cands = find_candidate(coords, pix=[None]*20, num=20,
                                   xa=xa, ya=0, xb=xb, yb=0,
                                   n=1, nx=1, ny=1, sumg=1,
                                   vpar=vpar, cpar=cpar, cal=calib)
    # expected candidates are those with x in [xa, xb]
    assert count == expected
    assert len(cands) == expected
    # check pnr indices correspond to list positions
    for cand in cands:
        assert xa - vpar.eps0 < coords[cand.pnr].x < xb + vpar.eps0
