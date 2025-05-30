import numpy as np
from typing import Tuple, List
from pyoptv.calibration import Calibration
from pyoptv.parameters import ControlPar, MMNP
from pyoptv.tracking_frame_buf import Target
from pyoptv.imgcoord import img_coord
from pyoptv.trafo import metric_to_pixel




def sortgrid(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    num: int,
    eps: float,
    pix: List[Target]
) -> List[Target]:
    """
    Literal translation of liboptv's sortgrid.c: 
    For each calibration point in fix, project to image, find nearest detected target in pix within eps,
    assign all fields, set pnr to i if found, else pnr=-999.
    """
    # Pre-allocate output array
    sorted_pix: List[Target] = [Target(-999, 0, 0, 0, 0, 0, 0, -1) for _ in range(nfix)]
    for i in range(nfix):
        # Project calibration point to image coordinates
        xp, yp = img_coord(fix[i], cal, cpar.mm)
        calib_point = metric_to_pixel(xp, yp, cpar)
        # If projected point is not touching the image border
        if (calib_point[0] > -eps and calib_point[1] > -eps and
            calib_point[0] < cpar.imx + eps and calib_point[1] < cpar.imy + eps):
            # Find the nearest target point
            j = nearest_neighbour_pix(pix, num, calib_point[0], calib_point[1], eps)
            if j != -999:
                # Assign all fields from pix[j], but set pnr to i
                sorted_pix[i] = Target(
                    pnr=i,
                    x=pix[j].x,
                    y=pix[j].y,
                    n=pix[j].n,
                    nx=pix[j].nx,
                    ny=pix[j].ny,
                    sumg=pix[j].sumg,
                    tnr=pix[j].tnr
                )
            # else: already set to unused
        # else: already set to unused
    return sorted_pix

def nearest_neighbour_pix(
    pix: List[Target],
    num: int,
    x: float,
    y: float,
    eps: float
) -> int:
    xmin, xmax = x - eps, x + eps
    ymin, ymax = y - eps, y + eps
    dmin, pnr = 1e20, -999

    for j in range(num):
        if ymin < pix[j].y < ymax and xmin < pix[j].x < xmax:
            d = np.sqrt((x - pix[j].x) ** 2 + (y - pix[j].y) ** 2)
            if d < dmin:
                dmin, pnr = d, j

    return pnr

def read_sortgrid_par(filename: str) -> int:
    try:
        with open(filename, 'r') as f:
            eps = int(f.readline().strip())
        return eps
    except Exception as e:
        print(f"Error reading sortgrid parameter from {filename}: {e}")
        return 0

def read_calblock(filename: str) -> Tuple[np.ndarray, int]:
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_points = len(lines)
        fix = np.empty(num_points, dtype=np.object)
        for i, line in enumerate(lines):
            parts = line.split()
            fix[i] = {'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3])}
        return fix, num_points
    except Exception as e:
        print(f"Can't open calibration block file: {filename}: {e}")
        return None, 0
