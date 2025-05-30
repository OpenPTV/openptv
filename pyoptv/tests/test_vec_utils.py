import pytest
import numpy as np
from pyoptv.vec_utils import (
    is_empty, norm, vec_init, vec_set, vec_copy, vec_subt, vec_add,
    vec_scalar_mul, vec_norm, vec_diff_norm, vec_dot, vec_cross,
    vec_cmp, vec_approx_cmp, unit_vector
)

def test_is_empty():
    assert is_empty(np.nan)
    assert not is_empty(1.0)

def test_norm():
    assert np.isclose(norm(1.0, 2.0, 2.0), 3.0)

def test_vec_init():
    vec = vec_init()
    assert np.isnan(vec[0]) and np.isnan(vec[1]) and np.isnan(vec[2])

def test_vec_set():
    vec = vec_set(1.0, 2.0, 3.0)
    assert np.allclose(vec, [1.0, 2.0, 3.0])

def test_vec_copy():
    src = np.array([1.0, 2.0, 3.0])
    dst = vec_copy(src)
    assert np.allclose(src, dst)
    assert src is not dst

def test_vec_subt():
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([0.5, 1.0, 1.5])
    result = vec_subt(vec1, vec2)
    assert np.allclose(result, [0.5, 1.0, 1.5])

def test_vec_add():
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([0.5, 1.0, 1.5])
    result = vec_add(vec1, vec2)
    assert np.allclose(result, [1.5, 3.0, 4.5])

def test_vec_scalar_mul():
    vec = np.array([1.0, 2.0, 3.0])
    result = vec_scalar_mul(vec, 2.0)
    assert np.allclose(result, [2.0, 4.0, 6.0])

def test_vec_norm():
    vec = np.array([1.0, 2.0, 2.0])
    assert np.isclose(vec_norm(vec), 3.0)

def test_vec_diff_norm():
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([0.0, 0.0, 0.0])
    assert np.isclose(vec_diff_norm(vec1, vec2), 3.7416573867739413)

def test_vec_dot():
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([0.5, 1.0, 1.5])
    assert np.isclose(vec_dot(vec1, vec2), 7.0)

def test_vec_cross():
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    result = vec_cross(vec1, vec2)
    assert np.allclose(result, [0.0, 0.0, 1.0])

def test_vec_cmp():
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([1.0, 2.0, 3.0])
    vec3 = np.array([0.0, 0.0, 0.0])
    assert vec_cmp(vec1, vec2)
    assert not vec_cmp(vec1, vec3)

def test_vec_approx_cmp():
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([1.0, 2.0, 3.0])
    vec3 = np.array([1.0, 2.0, 3.05])  # Adjusted value to be within 0.1 tolerance
    assert vec_approx_cmp(vec1, vec2, 1e-6)
    assert not vec_approx_cmp(vec1, vec3, 1e-6)
    assert vec_approx_cmp(vec1, vec3, 0.1)

def test_unit_vector():
    vec = np.array([1.0, 2.0, 2.0])
    result = unit_vector(vec)
    assert np.allclose(result, [0.33333333, 0.66666667, 0.66666667])

if __name__ == "__main__":
    pytest.main([__file__])
