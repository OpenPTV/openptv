import numpy as np
from typing import Tuple, Optional
from pyoptv.calibration import Calibration
from pyoptv.parameters import ControlPar, MMNP
from .multimed import trans_Cam_Point, multimed_nlay, back_trans_Point
from .trafo import flat_to_dist
from .vec_utils import vec_set


def flat_image_coord(
    orig_pos: Tuple[float, float, float],
    cal: Calibration,
    mm: MMNP
) -> Tuple[float, float]:
    """
    Calculates projection from coordinates in world space to metric coordinates in image space without distortions.
    Args:
        orig_pos: 3D position (X, Y, Z real space)
        cal: Camera calibration parameters
        mm: Layer thickness and refractive index parameters
    Returns:
        (x, y): metric coordinates of projection in the image space
    """
    # Prepare temporary calibration and variables
    cal_t = Calibration()
    cal_t.mmlut = cal.mmlut
    # Use the correct Python API for trans_Cam_Point
    ex_t, pos_t, cross_p, cross_c = trans_Cam_Point(
        cal.ext_par, mm, cal.glass_par, np.asarray(orig_pos)
    )
    cal_t.ext_par = ex_t
    X_t, Y_t = multimed_nlay(cal_t, mm, pos_t)
    pos_t = vec_set(X_t, Y_t, pos_t[2] if hasattr(pos_t, '__getitem__') else pos_t.z)
    pos = back_trans_Point(pos_t, mm, cal.glass_par, cross_p, cross_c)

    # Support both Vec3D and numpy array for pos
    if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
        dx = pos.x - cal.ext_par.x0
        dy = pos.y - cal.ext_par.y0
        dz = pos.z - cal.ext_par.z0
    else:
        dx = pos[0] - cal.ext_par.x0
        dy = pos[1] - cal.ext_par.y0
        dz = pos[2] - cal.ext_par.z0
    # Avoid division by zero in denominator
    deno = (
        cal.ext_par.dm[0][2] * dx
        + cal.ext_par.dm[1][2] * dy
        + cal.ext_par.dm[2][2] * dz
    )
    if np.isclose(deno, 0.0):
        return np.nan, np.nan
    x = -cal.int_par.cc * (
        cal.ext_par.dm[0][0] * dx
        + cal.ext_par.dm[1][0] * dy
        + cal.ext_par.dm[2][0] * dz
    ) / deno
    y = -cal.int_par.cc * (
        cal.ext_par.dm[0][1] * dx
        + cal.ext_par.dm[1][1] * dy
        + cal.ext_par.dm[2][1] * dz
    ) / deno
    return x, y


def img_coord(
    pos: Tuple[float, float, float],
    cal: Calibration,
    mm: MMNP
) -> Tuple[float, float]:
    """
    Uses flat_image_coord to estimate metric coordinates in image space from the 3D position in the world and distorts it using the Brown distortion model.
    Args:
        pos: 3D position (X, Y, Z real space)
        cal: Camera calibration parameters
        mm: Layer thickness and refractive index parameters
    Returns:
        (x, y): metric distorted coordinates of projection in the image space
    """
    x, y = flat_image_coord(pos, cal, mm)
    x, y = flat_to_dist(x, y, cal)
    return x, y


