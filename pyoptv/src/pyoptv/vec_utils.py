import numpy as np
from typing import Union

Vec2D = np.ndarray  # shape (2,)
Vec3D = np.ndarray  # shape (3,)

EMPTY_CELL: float = np.nan

def is_empty(val: float) -> bool:
    return np.isnan(val)

def norm(x: float, y: float, z: float) -> float:
    return np.sqrt(x * x + y * y + z * z)

def vec_init() -> Vec3D:
    return np.full(3, EMPTY_CELL, dtype=float)

def vec_set(x: float, y: float, z: float) -> Vec3D:
    return np.array([x, y, z], dtype=float)

def vec_copy(src: Vec3D) -> Vec3D:
    return np.copy(src)

def vec_subt(a: Vec3D, b: Vec3D) -> Vec3D:
    return a - b

def vec_add(a: Vec3D, b: Vec3D) -> Vec3D:
    return a + b

def vec_scalar_mul(a: Vec3D, scalar: float) -> Vec3D:
    return a * scalar

def vec_norm(a: Vec3D) -> float:
    return np.linalg.norm(a)

def vec_diff_norm(a: Vec3D, b: Vec3D) -> float:
    return np.linalg.norm(a - b)

def vec_dot(a: Vec3D, b: Vec3D) -> float:
    return float(np.dot(a, b))

def vec_cross(a: Vec3D, b: Vec3D) -> Vec3D:
    return np.cross(a, b)

def vec_cmp(a: Vec3D, b: Vec3D) -> bool:
    return np.array_equal(a, b)

def vec_approx_cmp(a: Vec3D, b: Vec3D, tol: float) -> bool:
    return np.allclose(a, b, atol=tol)

def unit_vector(a: Vec3D) -> Vec3D:
    n = np.linalg.norm(a)
    if n == 0:
        n = 1.0
    return a / n
