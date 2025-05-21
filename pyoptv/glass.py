import numpy as np
from numba import jit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Glass:
    def __init__(self, vec_x, vec_y, vec_z, n1, n2, n3, d):
        self.vec_x = vec_x
        self.vec_y = vec_y
        self.vec_z = vec_z
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.d = d

    def __repr__(self):
        return f"Glass(vec_x={self.vec_x}, vec_y={self.vec_y}, vec_z={self.vec_z}, n1={self.n1}, n2={self.n2}, n3={self.n3}, d={self.d})"

    @staticmethod
    @jit(nopython=True)
    def calculate_refraction_angle(n1, n2, angle_incidence):
        return np.arcsin(n1 / n2 * np.sin(angle_incidence))

    def optimize_refraction(self, initial_guess):
        def objective(x):
            angle_incidence = x[0]
            angle_refraction = self.calculate_refraction_angle(self.n1, self.n2, angle_incidence)
            return np.abs(angle_refraction - x[1])

        result = minimize(objective, initial_guess)
        return result.x

    def plot_refraction(self, angle_incidence):
        angle_refraction = self.calculate_refraction_angle(self.n1, self.n2, angle_incidence)
        plt.plot(angle_incidence, angle_refraction, 'o')
        plt.xlabel('Angle of Incidence (radians)')
        plt.ylabel('Angle of Refraction (radians)')
        plt.title('Refraction Angle')
        plt.show()
