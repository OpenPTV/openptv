import pytest
from pyoptv.epi import find_candidate, Candidate, Coord2D
from pyoptv.parameters import ControlPar, VolumePar
from pyoptv.calibration import Calibration

class DummyPix:
    n = 1
    nx = 1
    ny = 1
    sumg = 1

@pytest.fixture
def default_params():
    cpar = ControlPar(num_cams=1)
    vpar = VolumePar()
    cal = Calibration()
    return cpar, vpar, cal


def test_find_candidate_out_of_bounds(default_params):
    cpar, vpar, cal = default_params
    # Empty coordinate and pixel lists
    crd = []
    pix = []
    # Choose epipolar strip far outside the sensor extents
    xa, ya, xb, yb = 1000.0, 1000.0, 1001.0, 1001.0
    count, candidates = find_candidate(crd, pix, num=0,
                                       xa=xa, ya=ya, xb=xb, yb=yb,
                                       n=1, nx=1, ny=1, sumg=1,
                                       vpar=vpar, cpar=cpar, cal=cal)
    assert count == -1, "Out-of-bounds epipolar strip should return -1"
    assert candidates == [], "No candidates should be returned when out of bounds"


def test_find_candidate_in_bounds_no_hits(default_params):
    cpar, vpar, cal = default_params
    # Prepare a trivial single point list inside the sensor
    # Pixel size=0.01, image dims=256 -> extents: ~[-1.28,1.28]
    # Single point at (0,0)
    crd = [Coord2D(pnr=0, x=0.0, y=0.0)]
    # Pix entry must have n,nx,ny,sumg attributes
    pix = [DummyPix()]
    # Epipolar strip that includes (0,0) but no features on the line
    xa, ya, xb, yb = 0.0, 0.0, 1.0, 1.0
    count, candidates = find_candidate(crd, pix, num=1,
                                       xa=xa, ya=ya, xb=xb, yb=yb,
                                       n=1, nx=1, ny=1, sumg=1,
                                       vpar=vpar, cpar=cpar, cal=cal)
    # With tol_band_width=0, only exact matches on the line get through
    # Our DummyPix has minimal sumg ratios, so we expect zero candidates
    assert count == 0, "No candidates should be returned when no hits found"
    assert candidates == [], "Empty candidate list expected"
