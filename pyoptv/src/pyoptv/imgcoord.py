import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def flat_image_coord(pos, cal, mm):
    # Accepts cal as a Calibration object
    deno = (cal.ext_par.dm[0][2] * (pos[0] - cal.ext_par.x0) +
            cal.ext_par.dm[1][2] * (pos[1] - cal.ext_par.y0) +
            cal.ext_par.dm[2][2] * (pos[2] - cal.ext_par.z0))

    x = -cal.int_par.cc * (cal.ext_par.dm[0][0] * (pos[0] - cal.ext_par.x0) +
                           cal.ext_par.dm[1][0] * (pos[1] - cal.ext_par.y0) +
                           cal.ext_par.dm[2][0] * (pos[2] - cal.ext_par.z0)) / deno

    y = -cal.int_par.cc * (cal.ext_par.dm[0][1] * (pos[0] - cal.ext_par.x0) +
                           cal.ext_par.dm[1][1] * (pos[1] - cal.ext_par.y0) +
                           cal.ext_par.dm[2][1] * (pos[2] - cal.ext_par.z0)) / deno

    return x, y

def img_coord(pos, cal, mm):
    x, y = flat_image_coord(pos, cal, mm)
    x, y = flat_to_dist(x, y, cal)
    return x, y

def flat_to_dist(x, y, cal):
    r = np.sqrt(x**2 + y**2)
    x_dist = x * (1 + cal.dist_par.k1 * r**2 + cal.dist_par.k2 * r**4 + cal.dist_par.k3 * r**6)
    y_dist = y * (1 + cal.dist_par.k1 * r**2 + cal.dist_par.k2 * r**4 + cal.dist_par.k3 * r**6)
    return x_dist, y_dist
def flat_image_coord_numba(pos, cal, mm):
    deno = (cal['ext_par']['dm'][0][2] * (pos[0] - cal['ext_par']['x0']) +
            cal['ext_par']['dm'][1][2] * (pos[1] - cal['ext_par']['y0']) +
            cal['ext_par']['dm'][2][2] * (pos[2] - cal['ext_par']['z0']))

    x = -cal['int_par']['cc'] * (cal['ext_par']['dm'][0][0] * (pos[0] - cal['ext_par']['x0']) +
                                 cal['ext_par']['dm'][1][0] * (pos[1] - cal['ext_par']['y0']) +
                                 cal['ext_par']['dm'][2][0] * (pos[2] - cal['ext_par']['z0'])) / deno

    y = -cal['int_par']['cc'] * (cal['ext_par']['dm'][0][1] * (pos[0] - cal['ext_par']['x0']) +
                                 cal['ext_par']['dm'][1][1] * (pos[1] - cal['ext_par']['y0']) +
                                 cal['ext_par']['dm'][2][1] * (pos[2] - cal['ext_par']['z0'])) / deno

    return x, y
def img_coord_numba(pos, cal, mm):
    x, y = flat_image_coord_numba(pos, cal, mm)
    x, y = flat_to_dist_numba(x, y, cal)
    return x, y
def flat_to_dist_numba(x, y, cal):
    r = np.sqrt(x**2 + y**2)
    x_dist = x * (1 + cal['dist_par']['k1'] * r**2 + cal['dist_par']['k2'] * r**4 + cal['dist_par']['k3'] * r**6)
    y_dist = y * (1 + cal['dist_par']['k1'] * r**2 + cal['dist_par']['k2'] * r**4 + cal['dist_par']['k3'] * r**6)
    return x_dist, y_dist

def plot_image_coords(pos, cal, mm):
    x, y = img_coord(pos, cal, mm)
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Image Coordinates')
    plt.show()
