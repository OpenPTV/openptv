import pytest
import numpy as np
from pyoptv.glass import Glass

def test_glass_initialization():
    glass = Glass(0.1, 0.2, 0.3, 1.0, 1.5, 1.33, 0.5)
    assert glass.vec_x == 0.1
    assert glass.vec_y == 0.2
    assert glass.vec_z == 0.3
    assert glass.n1 == 1.0
    assert glass.n2 == 1.5
    assert glass.n3 == 1.33
    assert glass.d == 0.5

def test_calculate_refraction_angle():
    angle_incidence = np.pi / 4
    angle_refraction = Glass.calculate_refraction_angle(1.0, 1.5, angle_incidence)
    expected_angle_refraction = np.arcsin(1.0 / 1.5 * np.sin(angle_incidence))
    assert np.isclose(angle_refraction, expected_angle_refraction)

def test_optimize_refraction():
    glass = Glass(0.1, 0.2, 0.3, 1.0, 1.5, 1.33, 0.5)
    initial_guess = [np.pi / 4, np.pi / 6]
    optimized_angles = glass.optimize_refraction(initial_guess)
    assert len(optimized_angles) == 2

def test_plot_refraction():
    glass = Glass(0.1, 0.2, 0.3, 1.0, 1.5, 1.33, 0.5)
    angle_incidence = np.linspace(0, np.pi / 2, 100)
    glass.plot_refraction(angle_incidence)
