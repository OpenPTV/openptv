import math
import struct
from typing import List

from .vec_utils import *

def test_vec_init():
    """tests vec_init
    """
    init = [0.0, 0.0, 0.0]
    vec_init(init)
    for i in range(3):
        assert math.isnan(init[i])
        
def test_vec_set():
    dest = [0.0, 0.0, 0.0]
    vec_set(dest, 1.2, 3.4, 5.6)
    assert dest == [1.2, 3.4, 5.6]
    
def test_vec_copy():
    dest = [0.0, 0.0, 0.0]
    src = [1.2, 3.4, 5.6]
    vec_copy(dest, src)
    assert dest == src
    
def test_vec_subt():
    from_ = [1.0, 2.0, 3.0]
    sub = [4.0, 5.0, 6.0]
    output = [0.0, 0.0, 0.0]
    vec_subt(from_, sub, output)
    assert output == [-3.0, -3.0, -3.0]
    
def test_vec_add():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    output = [0.0, 0.0, 0.0]
    vec_add(vec1, vec2, output)
    assert output == [5.0, 7.0, 9.0]
    
def test_vec_scalar_mul():
    vec = [1.0, 2.0, 3.0]
    scalar = 2.0
    output = [0.0, 0.0, 0.0]
    vec_scalar_mul(vec, scalar, output)
    assert output == [2.0, 4.0, 6.0]
    
def test_vec_diff_norm():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    assert math.isclose(vec_diff_norm(vec1, vec2), 5.1962, rel_tol=1e-4)
    
def test_vec_norm():
    vec = [1.0, 2.0, 3.0]
    assert math.isclose(vec_norm(vec), 3.7416, rel_tol=1e-4)
    
def test_vec_dot():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    assert math.isclose(vec_dot(vec1, vec2), 32.0, rel_tol=1e-4)
    
def test_vec_cross():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    output = [0.0, 0.0, 0.0]
    vec_cross(vec1, vec2, output)
    assert output == [-3.0, 6.0, -3.0]
    
def test_vec_cmp():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]
    assert vec_cmp(vec1, vec2, 1e-4)
    vec3 = [4.0, 5.0, 6.0]
    assert not vec_cmp(vec1, vec3, 1e-4)
    
def test_vec_approx_cmp():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.1, 1.9, 3.1]
    eps = 0.2
    assert vec_approx_cmp(vec1, vec2, eps)
    
def test_unit_vector():
    vec = [1.0, 2.0, 3.0]
    out = [0.0, 0.0, 0.0]
    unit_vector(vec, out)
    assert math.isclose(vec_norm(out), 1.0, rel_tol=1e-4)
    
def test_vec_scalar_mul():
    vec = [1.0, 2.0, 3.0]
    scalar = 2.0
    output = [0.0, 0.0, 0.0]
    vec_scalar_mul(vec, scalar, output)
    assert output == [2.0, 4.0, 6.0]