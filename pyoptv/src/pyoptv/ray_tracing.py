import numpy as np
from .vec_utils import vec_set, vec_subt, vec_add, vec_scalar_mul, vec_norm, vec_dot, unit_vector
from pyoptv.calibration import Calibration
from pyoptv.parameters import MMNP
from .lsqadj import matmul

def ray_tracing(x: float, y: float, cal: Calibration, mm: MMNP) -> tuple[np.ndarray, np.ndarray]:
    """
    Translated implementation of ray tracing from C to Python.
    Args:
        x: X coordinate in image space
        y: Y coordinate in image space
        cal: Camera calibration parameters
        mm: Multi-media parameters (thickness and refractive indices)

    Returns:
        X: Intersection point in the glass
        out: Direction vector after passing through the glass
    """


    # Initial ray direction in global coordinate system
    tmp1 = vec_set(x, y, -1 * cal.int_par.cc)
    tmp1 = unit_vector(tmp1)
    # Reshape tmp1 to 2D array for matmul
    tmp1 = tmp1.reshape(-1, 1)
    start_dir = matmul(cal.ext_par.dm, tmp1, 3, 3, 1, 3, 3).flatten()

    primary_point = vec_set(cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0)

    glass_dir = vec_set(cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z)
    glass_dir = unit_vector(glass_dir)
    c = vec_norm(tmp1) + mm.d[0]

    dist_cam_glass = vec_dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / vec_dot(glass_dir, start_dir)
    tmp1 = vec_scalar_mul(start_dir, d1)
    Xb = vec_add(primary_point, tmp1)

    n = vec_dot(start_dir, glass_dir)
    tmp1 = vec_scalar_mul(glass_dir, n)
    tmp2 = vec_subt(start_dir, tmp1)
    bp = unit_vector(tmp2)

    # Align Snell's law computation with C implementation
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]
    n = -1*np.sqrt(1 - p * p)

    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n)
    a2 = vec_add(tmp1, tmp2)

    d2 = mm.d[0] / abs(vec_dot(glass_dir, a2))

    tmp1 = vec_scalar_mul(a2, d2)
    X = vec_add(Xb, tmp1)

    n = vec_dot(a2, glass_dir)
    tmp2 = vec_subt(a2, tmp2)
    bp = unit_vector(tmp2)

    p = np.sqrt(1 - n * n)
    p = p * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)

    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n)
    out = vec_add(tmp1, tmp2)

    return X, out
