import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .vec_utils import vec_set, vec_subt, vec_add, vec_scalar_mul, vec_norm, vec_dot, unit_vector
from pyoptv.calibration import Calibration
from pyoptv.parameters import MMNP

def ray_tracing(
    x: float,
    y: float,
    cal: Calibration,
    mm: MMNP,
    X: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    C-style: ray_tracing(x, y, cal, mm, X, a) modifies X, a in-place
    Pythonic: ray_tracing(x, y, cal, mm) -> (X, a)
    """
    if X is not None and a is not None:
        _ray_tracing_impl(x, y, cal, mm, X, a)
        return None
    else:
        X = np.zeros(3)
        a = np.zeros(3)
        _ray_tracing_impl(x, y, cal, mm, X, a)
        return X, a

# The C-like implementation for internal use
def _ray_tracing_impl(
    x: float,
    y: float,
    cal: Calibration,
    mm: MMNP,
    X: np.ndarray,
    a: np.ndarray
) -> None:
    tmp1 = vec_set(x, y, -1 * cal.int_par.cc)
    tmp1 = unit_vector(tmp1)
    start_dir = np.dot(cal.ext_par.dm, tmp1)
    primary_point = vec_set(cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0)
    glass_dir = vec_set(cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z)
    glass_dir = unit_vector(glass_dir)
    c = vec_norm(glass_dir) + mm.d[0]
    dist_cam_glass = vec_dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / vec_dot(glass_dir, start_dir)
    tmp2 = vec_scalar_mul(start_dir, d1)
    Xb = vec_add(primary_point, tmp2)
    n = vec_dot(start_dir, glass_dir)
    tmp2 = vec_scalar_mul(glass_dir, n)
    bp = vec_subt(start_dir, tmp2)
    bp = unit_vector(bp)
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]
    n = -np.sqrt(1 - p * p)
    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n)
    a2 = vec_add(tmp1, tmp2)
    d2 = mm.d[0] / abs(vec_dot(glass_dir, a2))
    tmp1 = vec_scalar_mul(a2, d2)
    X[:] = vec_add(Xb, tmp1)
    n = vec_dot(a2, glass_dir)
    tmp2 = vec_scalar_mul(glass_dir, n)
    bp = vec_subt(a2, tmp2)
    bp = unit_vector(bp)
    p = np.sqrt(1 - n * n) * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)
    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n)
    a[:] = vec_add(tmp1, tmp2)
