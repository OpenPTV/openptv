import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

EMPTY_CELL = np.nan

def is_empty(x):
    return np.isnan(x)
def norm(x, y, z):
    return np.sqrt(x * x + y * y + z * z)
def vec_init():
    return np.full(3, EMPTY_CELL)
def vec_set(x, y, z):
    return np.array([x, y, z])
def vec_copy(src):
    return np.copy(src)
def vec_subt(from_vec, sub_vec):
    return from_vec - sub_vec
def vec_add(vec1, vec2):
    return vec1 + vec2
def vec_scalar_mul(vec, scalar):
    return vec * scalar
def vec_norm(vec):
    return norm(vec[0], vec[1], vec[2])
def vec_diff_norm(vec1, vec2):
    return norm(vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2])
def vec_dot(vec1, vec2):
    return np.dot(vec1, vec2)
def vec_cross(vec1, vec2):
    return np.cross(vec1, vec2)
def vec_cmp(vec1, vec2):
    return np.array_equal(vec1, vec2)
def vec_approx_cmp(vec1, vec2, eps):
    return np.allclose(vec1, vec2, atol=eps)
def unit_vector(vec):
    normed = vec_norm(vec)
    if normed == 0:
        normed = 1.0
    return vec_scalar_mul(vec, 1.0 / normed)
