import pytest
import numpy as np
from pyoptv.lsqadj import ata, atl, matinv, matmul, norm_cross

def test_ata():
    a = np.array([[1, 0, 1], [2, 2, 4], [1, 2, 3], [2, 4, 3]])
    expected = np.array([[10, 14, 18], [14, 24, 26], [18, 26, 35]])
    result = ata(a, 4, 3, 3)
    assert np.allclose(result, expected)

def test_atl():
    a = np.array([[1, 0, 1], [2, 2, 4], [1, 2, 3], [2, 4, 3]])
    l = np.array([1, 2, 3, 4])
    expected = np.array([16, 26, 30])
    result = atl(a, l, 4, 3, 3)
    assert np.allclose(result, expected)

def test_matinv():
    a = np.array([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
    expected = np.array([[1.090909, -0.545455, -0.090909], [0.227273, 0.136364, -0.227273], [-0.181818, 0.090909, 0.181818]])
    result = matinv(a, 3, 3)
    assert np.allclose(result, expected)

def test_matmul():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[10, 11], [12, 13], [14, 15]])
    expected = np.array([[76, 82], [184, 199], [292, 316]])
    result = matmul(a, b, 3, 3, 2, 3, 3)
    assert np.allclose(result, expected)

def test_norm_cross():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    expected = np.array([-3, 6, -3])
    result = norm_cross(a, b)
    assert np.allclose(result, expected)
