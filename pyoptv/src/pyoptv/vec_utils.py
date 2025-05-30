import numpy as np
from typing import Union

class Vec2D:
    x: float
    y: float
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    def __getitem__(self, idx: int) -> float:
        return (self.x, self.y)[idx]
    def __iter__(self) -> 'Vec2D':
        return iter((self.x, self.y))
    def __add__(self, other: 'Vec2D') -> 'Vec2D':
        return Vec2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other: 'Vec2D') -> 'Vec2D':
        return Vec2D(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar: float) -> 'Vec2D':
        return Vec2D(self.x * scalar, self.y * scalar)
    def __rmul__(self, scalar: float) -> 'Vec2D':
        return self.__mul__(scalar)

class Vec3D:
    x: float
    y: float
    z: float
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z
    def __getitem__(self, idx: int) -> float:
        return (self.x, self.y, self.z)[idx]
    def __setitem__(self, idx: int, value: float) -> None:
        if idx == 0:
            self.x = value
        elif idx == 1:
            self.y = value
        elif idx == 2:
            self.z = value
        else:
            raise IndexError('Vec3D index out of range')
    def __iter__(self) -> 'Vec3D':
        return iter((self.x, self.y, self.z))
    def __add__(self, other: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, scalar: float) -> 'Vec3D':
        return Vec3D(self.x * scalar, self.y * scalar, self.z * scalar)
    def __rmul__(self, scalar: float) -> 'Vec3D':
        return self.__mul__(scalar)

EMPTY_CELL: float = np.nan

def is_empty(x: float) -> bool:
    return np.isnan(x)

def norm(x: float, y: float, z: float) -> float:
    return np.sqrt(x * x + y * y + z * z)

def vec_init() -> Vec3D:
    return Vec3D(EMPTY_CELL, EMPTY_CELL, EMPTY_CELL)

def vec_set(x: float, y: float, z: float) -> Vec3D:
    return Vec3D(x, y, z)

def vec_copy(src: Vec3D) -> Vec3D:
    return Vec3D(src.x, src.y, src.z)

def vec_subt(from_vec: Vec3D, sub_vec: Vec3D) -> Vec3D:
    return from_vec - sub_vec

def vec_add(vec1: Vec3D, vec2: Vec3D) -> Vec3D:
    return vec1 + vec2

def vec_scalar_mul(vec: Vec3D, scalar: float) -> Vec3D:
    return vec * scalar

def vec_norm(vec: Vec3D) -> float:
    return norm(vec.x, vec.y, vec.z)

def vec_diff_norm(vec1: Vec3D, vec2: Vec3D) -> float:
    return norm(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z)

def vec_dot(vec1: Vec3D, vec2: Vec3D) -> float:
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z

def vec_cross(vec1: Vec3D, vec2: Vec3D) -> Vec3D:
    return Vec3D(
        vec1.y * vec2.z - vec1.z * vec2.y,
        vec1.z * vec2.x - vec1.x * vec2.z,
        vec1.x * vec2.y - vec1.y * vec2.x
    )

def vec_cmp(vec1: Vec3D, vec2: Vec3D) -> bool:
    return (vec1.x == vec2.x and vec1.y == vec2.y and vec1.z == vec2.z)

def vec_approx_cmp(vec1: Vec3D, vec2: Vec3D, eps: float) -> bool:
    return (abs(vec1.x - vec2.x) <= eps and abs(vec1.y - vec2.y) <= eps and abs(vec1.z - vec2.z) <= eps)

def unit_vector(vec: Vec3D) -> Vec3D:
    n = vec_norm(vec)
    if n == 0:
        n = 1.0
    return vec_scalar_mul(vec, 1.0 / n)
