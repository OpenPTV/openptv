import numpy as np
import numba
from scipy.optimize import minimize
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def ray_tracing(x, y, cal, mm, X, a):
    tmp1 = np.array([x, y, -1 * cal.int_par.cc])
    tmp1 /= np.linalg.norm(tmp1)
    start_dir = np.dot(cal.ext_par.dm, tmp1)
    primary_point = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
    glass_dir = np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
    glass_dir /= np.linalg.norm(glass_dir)
    c = np.linalg.norm(glass_dir) + mm.d[0]
    dist_cam_glass = np.dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / np.dot(glass_dir, start_dir)
    Xb = primary_point + d1 * start_dir
    n = np.dot(start_dir, glass_dir)
    bp = start_dir - n * glass_dir
    bp /= np.linalg.norm(bp)
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]
    n = -np.sqrt(1 - p * p)
    a2 = p * bp + n * glass_dir
    d2 = mm.d[0] / abs(np.dot(glass_dir, a2))
    X[:] = Xb + d2 * a2
    n = np.dot(a2, glass_dir)
    bp = a2 - n * glass_dir
    bp /= np.linalg.norm(bp)
    p = np.sqrt(1 - n * n) * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)
    a[:] = p * bp + n * glass_dir
