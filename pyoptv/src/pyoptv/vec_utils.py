import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, idx):
        return (self.x, self.y)[idx]
    def __setitem__(self, idx, value):
        if idx == 0:
            self.x = value
        elif idx == 1:
            self.y = value
        else:
            raise IndexError('Vec2D index out of range')
    def __iter__(self):
        return iter((self.x, self.y))
    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar):
        return Vec2D(self.x * scalar, self.y * scalar)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

class Vec3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __getitem__(self, idx):
        return (self.x, self.y, self.z)[idx]
    def __setitem__(self, idx, value):
        if idx == 0:
            self.x = value
        elif idx == 1:
            self.y = value
        elif idx == 2:
            self.z = value
        else:
            raise IndexError('Vec3D index out of range')
    def __iter__(self):
        return iter((self.x, self.y, self.z))
    def __add__(self, other):
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, scalar):
        return Vec3D(self.x * scalar, self.y * scalar, self.z * scalar)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

# Math utilities updated for Vec2D/Vec3D
EMPTY_CELL = np.nan

def is_empty(x):
    return np.isnan(x)
def norm(x, y, z):
    return np.sqrt(x * x + y * y + z * z)
def vec_init():
    return Vec3D(EMPTY_CELL, EMPTY_CELL, EMPTY_CELL)
def vec_set(x, y, z):
    return Vec3D(x, y, z)
def vec_copy(src):
    return Vec3D(src.x, src.y, src.z)
def vec_subt(from_vec, sub_vec):
    return from_vec - sub_vec
def vec_add(vec1, vec2):
    return vec1 + vec2
def vec_scalar_mul(vec, scalar):
    return vec * scalar
def vec_norm(vec):
    return norm(vec.x, vec.y, vec.z)
def vec_diff_norm(vec1, vec2):
    return norm(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z)
def vec_dot(vec1, vec2):
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z
def vec_cross(vec1, vec2):
    return Vec3D(
        vec1.y * vec2.z - vec1.z * vec2.y,
        vec1.z * vec2.x - vec1.x * vec2.z,
        vec1.x * vec2.y - vec1.y * vec2.x
    )
def vec_cmp(vec1, vec2):
    return (vec1.x == vec2.x and vec1.y == vec2.y and vec1.z == vec2.z)
def vec_approx_cmp(vec1, vec2, eps):
    return (abs(vec1.x - vec2.x) <= eps and abs(vec1.y - vec2.y) <= eps and abs(vec1.z - vec2.z) <= eps)
def unit_vector(vec):
    n = vec_norm(vec)
    if n == 0:
        n = 1.0
    return vec_scalar_mul(vec, 1.0 / n)
