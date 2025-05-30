import numpy as np
from .vec_utils import vec_set, vec_subt, vec_add, vec_scalar_mul, vec_norm, vec_dot, unit_vector
from pyoptv.calibration import Calibration
from pyoptv.parameters import MMNP

def ray_tracing(x: float, 
                y: float, 
                cal: Calibration, 
                mm: MMNP
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal implementation of ray tracing.
    Args:
        x: X coordinate in image space
        y: Y coordinate in image space
        cal: Camera calibration parameters
        mm: Multi-media parameters (thickness and refractive indices)
        
    Output:
        X: Output array for the intersection point in the glass
        out: Output array for the direction after passing through the glass
    """

    # Initial ray direction in global coordinate system
    tmp1 = np.array([x, y, -1 * cal.int_par.cc])
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    start_dir = np.dot(cal.ext_par.dm, tmp1)
    primary_point = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
    glass_dir = np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
    glass_dir = glass_dir / np.linalg.norm(glass_dir)
    c = np.linalg.norm(glass_dir) + mm.d[0]
    dist_cam_glass = np.dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / np.dot(glass_dir, start_dir)
    tmp1 = start_dir * d1
    Xb = primary_point + tmp1
    n = np.dot(start_dir, glass_dir)
    tmp1 = glass_dir * n
    tmp2 = start_dir - tmp1
    bp = tmp2 / np.linalg.norm(tmp2)
    
    # Transform to direction inside glass, using Snell's law
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]
    n = -np.sqrt(1 - p * p)
    tmp1 = bp * p
    tmp2 = glass_dir * n
    a2 = tmp1 + tmp2
    d2 = mm.d[0] / abs(np.dot(glass_dir, a2))
    tmp1 = a2 * d2
    X = Xb + tmp1

    # Again, direction in next medium
    n = np.dot(a2, glass_dir)
    tmp2 = glass_dir * n
    tmp3 = a2 - tmp2
    bp = tmp3 / np.linalg.norm(tmp3)
    p = np.sqrt(1 - n * n) * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)
    tmp1 = bp * p
    tmp2 = glass_dir * n
    out = tmp1 + tmp2

    return X, out
