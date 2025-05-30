import numpy as np
from typing import Union

class Vec2D:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __add__(self, other: 'Vec2D') -> 'Vec2D':
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vec2D') -> 'Vec2D':
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vec2D':
        return Vec2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vec2D':
        return self.__mul__(scalar)

class Vec3D:
    def __init__(self, x: float, y: float, z: float):
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __add__(self, other: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3D':
        return Vec3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vec3D':
        return self.__mul__(scalar)

EMPTY_CELL: float = np.nan

def is_empty(val: float) -> bool:
    return np.isnan(val)

def norm(x: float, y: float, z: float) -> float:
    return np.sqrt(x * x + y * y + z * z)

def vec_init() -> Vec3D:
    return Vec3D(EMPTY_CELL, EMPTY_CELL, EMPTY_CELL)

def vec_set(x: float, y: float, z: float) -> Vec3D:
    return Vec3D(x, y, z)

def vec_copy(src: Vec3D) -> Vec3D:
    return Vec3D(src.x, src.y, src.z)

def vec_subt(a: Vec3D, b: Vec3D) -> Vec3D:
    return a - b

def vec_add(a: Vec3D, b: Vec3D) -> Vec3D:
    return a + b

def vec_scalar_mul(a: Vec3D, scalar: float) -> Vec3D:
    return a * scalar

def vec_norm(a: Vec3D) -> float:
    return norm(a.x, a.y, a.z)

def vec_diff_norm(a: Vec3D, b: Vec3D) -> float:
    return norm(a.x - b.x, a.y - b.y, a.z - b.z)

def vec_dot(a: Vec3D, b: Vec3D) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def vec_cross(a: Vec3D, b: Vec3D) -> Vec3D:
    return Vec3D(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )

def vec_cmp(a: Vec3D, b: Vec3D) -> bool:
    return (a.x == b.x and a.y == b.y and a.z == b.z)

def vec_approx_cmp(a: Vec3D, b: Vec3D, tol: float) -> bool:
    return (abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol and abs(a.z - b.z) <= tol)

def unit_vector(a: Vec3D) -> Vec3D:
    n = vec_norm(a)
    if n == 0:
        n = 1.0
    return vec_scalar_mul(a, 1.0 / n)
