import pytest
from pyoptv.epi import find_candidate, Coord2D
from pyoptv.parameters import ControlPar, VolumePar
from pyoptv.calibration import Calibration

def make_dummy_list(num, xmin, xmax, ymin, ymax):
    # generate a sorted list of Coord2D points along x with y between bounds
    pts = []
    step = (xmax - xmin) / (num - 1) if num > 1 else 0
    for i in range(num):
        x = xmin + i * step
        y = (ymin + ymax) / 2
        pt = Coord2D(pnr=i, x=x, y=y)
        pts.append(pt)
    return pts

@pytest.fixture
def dummy_params():
    cpar = ControlPar()
    # set image pixel extents and size
    cpar.pix_x = 1.0; cpar.pix_y = 1.0; cpar.imx = 100; cpar.imy = 100
    vpar = VolumePar(); vpar.eps0 = 0.1; vpar.cn = vpar.cnx = vpar.cny = 0; vpar.csumg = -1
    calib = Calibration(); calib.int_par.xh = calib.int_par.yh = 0
    return cpar, vpar, calib


def test_out_of_bounds_returns_negative(dummy_params):
    cpar, vpar, calib = dummy_params
    coords = make_dummy_list(10, 0, 1, 0, 1)
    pix = [None] * 10
    # epipolar line entirely outside sensor region
    xa, ya, xb, yb = -200, -200, -150, -150
    count, cands = find_candidate(coords, pix, len(coords), xa, ya, xb, yb,
                                   n=0, nx=0, ny=0, sumg=0,
                                   vpar=vpar, cpar=cpar, cal=calib)
    assert count == -1
    assert cands == []

@pytest.mark.parametrize("xa,xb,expected", [
    (0, 10, 11),  # endpoints inclusive with tol
    (5, 15, 11),
    (10, 20, 1),
])
def test_in_bounds_candidate_counts(dummy_params, xa, xb, expected):
    cpar, vpar, calib = dummy_params
    # single horizontal epipolar at y=0
    coords = make_dummy_list(20, 0, 20, 0, 0)
    pix = [type('T', (), {'n':1, 'nx':1, 'ny':1, 'sumg':1})() for _ in range(20)]
    count, cands = find_candidate(coords, pix, len(coords), xa, 0, xb, 0,
                                   n=1, nx=1, ny=1, sumg=1,
                                   vpar=vpar, cpar=cpar, cal=calib)
    assert count == expected
    assert len(cands) == expected
    for cand in cands:
        x = coords[cand.pnr].x
        assert xa - vpar.eps0 <= x <= xb + vpar.eps0
