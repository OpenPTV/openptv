import numpy as np
import numba
from scipy.optimize import minimize
import matplotlib.pyplot as plt

EMPTY_CELL = np.nan

@numba.jit(nopython=True)
def is_empty(x):
    return np.isnan(x)

@numba.jit(nopython=True)
def norm(x, y, z):
    return np.sqrt(x * x + y * y + z * z)

@numba.jit(nopython=True)
def vec_init():
    return np.full(3, EMPTY_CELL)

@numba.jit(nopython=True)
def vec_set(x, y, z):
    return np.array([x, y, z])

@numba.jit(nopython=True)
def vec_copy(src):
    return np.copy(src)

@numba.jit(nopython=True)
def vec_subt(from_vec, sub_vec):
    return from_vec - sub_vec

@numba.jit(nopython=True)
def vec_add(vec1, vec2):
    return vec1 + vec2

@numba.jit(nopython=True)
def vec_scalar_mul(vec, scalar):
    return vec * scalar

@numba.jit(nopython=True)
def vec_norm(vec):
    return norm(vec[0], vec[1], vec[2])

@numba.jit(nopython=True)
def vec_diff_norm(vec1, vec2):
    return norm(vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2])

@numba.jit(nopython=True)
def vec_dot(vec1, vec2):
    return np.dot(vec1, vec2)

@numba.jit(nopython=True)
def vec_cross(vec1, vec2):
    return np.cross(vec1, vec2)

@numba.jit(nopython=True)
def vec_cmp(vec1, vec2):
    return np.array_equal(vec1, vec2)

@numba.jit(nopython=True)
def vec_approx_cmp(vec1, vec2, eps):
    return np.allclose(vec1, vec2, atol=eps)

@numba.jit(nopython=True)
def unit_vector(vec):
    normed = vec_norm(vec)
    if normed == 0:
        normed = 1.0
    return vec_scalar_mul(vec, 1.0 / normed)
