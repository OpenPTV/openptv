import pytest
import numpy as np
from pyoptv import epi

def test_epi_function1():
    # Test for epi_function1
    result = epi.epi_function1(np.array([1, 2, 3]))
    expected = np.array([1, 4, 9])
    np.testing.assert_array_equal(result, expected)

def test_epi_function2():
    # Test for epi_function2
    result = epi.epi_function2(np.array([1, 2, 3]), np.array([4, 5, 6]))
    expected = np.array([5, 7, 9])
    np.testing.assert_array_equal(result, expected)

def test_epi_function3():
    # Test for epi_function3
    result = epi.epi_function3(np.array([1, 2, 3]), 2)
    expected = np.array([2, 4, 6])
    np.testing.assert_array_equal(result, expected)
