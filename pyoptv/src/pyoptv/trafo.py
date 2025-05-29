import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ControlPar:
    def __init__(self, imx, imy, pix_x, pix_y, chfield):
        self.imx = imx
        self.imy = imy
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.chfield = chfield

class Calibration:
    def __init__(self, xh, yh, added_par):
        self.xh = xh
        self.yh = yh
        self.added_par = added_par

class AddedPar:
    def __init__(self, k1, k2, k3, p1, p2, scx, she):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.scx = scx
        self.she = she
def old_pixel_to_metric(x_pixel, y_pixel, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode):
    if y_remap_mode == 1:
        y_pixel = 2.0 * y_pixel + 1.0
    elif y_remap_mode == 2:
        y_pixel *= 2.0

    x_metric = (x_pixel - im_size_x / 2.0) * pix_size_x
    y_metric = (im_size_y / 2.0 - y_pixel) * pix_size_y

    return x_metric, y_metric

def pixel_to_metric(x_pixel, y_pixel, parameters):
    return old_pixel_to_metric(x_pixel, y_pixel, parameters.imx, parameters.imy, parameters.pix_x, parameters.pix_y, parameters.chfield)
def old_metric_to_pixel(x_metric, y_metric, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode):
    x_pixel = (x_metric / pix_size_x) + (im_size_x / 2.0)
    y_pixel = (im_size_y / 2.0) - (y_metric / pix_size_y)

    if y_remap_mode == 1:
        y_pixel = (y_pixel - 1.0) / 2.0
    elif y_remap_mode == 2:
        y_pixel /= 2.0

    return x_pixel, y_pixel

def metric_to_pixel(x_metric, y_metric, parameters):
    return old_metric_to_pixel(x_metric, y_metric, parameters.imx, parameters.imy, parameters.pix_x, parameters.pix_y, parameters.chfield)
def distort_brown_affin(x, y, ap):
    r = np.sqrt(x * x + y * y)
    if r < 1e-10:
        return 0.0, 0.0

    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    radial_factor = 1.0 + ap.k1 * r2 + ap.k2 * r4 + ap.k3 * r6

    x_dist = x * radial_factor + ap.p1 * (r2 + 2 * x * x) + 2 * ap.p2 * x * y
    y_dist = y * radial_factor + ap.p2 * (r2 + 2 * y * y) + 2 * ap.p1 * x * y

    sin_she = np.sin(ap.she)
    cos_she = np.cos(ap.she)

    x1 = ap.scx * (x_dist - sin_she * y_dist)
    y1 = ap.scx * cos_she * y_dist

    return x1, y1

def correct_brown_affin(x, y, ap):
    sin_she = np.sin(ap.she)
    cos_she = np.cos(ap.she)
    inv_scx = 1.0 / ap.scx

    xq = x * inv_scx
    yq = y * inv_scx / cos_she
    xq += yq * sin_she

    MAX_ITER = 20
    DAMPING = 0.7
    TOL = 1e-8

    for _ in range(MAX_ITER):
        xq_old = xq
        yq_old = yq

        xt, yt = distort_brown_affin(xq, yq, ap)

        dx = (x - xt) * inv_scx
        dy = (y - yt) * inv_scx

        xq += dx * DAMPING
        yq += dy * DAMPING

        change = np.sqrt((xq - xq_old) ** 2 + (yq - yq_old) ** 2)
        pos_magnitude = np.sqrt(xq * xq + yq * yq)
        if pos_magnitude > 1e-10 and change / pos_magnitude < TOL:
            break

    return xq, yq

def correct_brown_affine_exact(x, y, ap, tol):
    r_init = np.sqrt(x * x + y * y)
    if r_init < 1e-10:
        return 0.0, 0.0

    sin_she = np.sin(ap.she)
    cos_she = np.cos(ap.she)
    inv_scx = 1.0 / ap.scx

    xq = (x + y * sin_she) * inv_scx
    yq = y / cos_she

    MAX_ITER = 50
    DAMPING = 0.5

    for _ in range(MAX_ITER):
        r2 = xq * xq + yq * yq
        r4 = r2 * r2
        r6 = r4 * r2

        radial_factor = ap.k1 * r2 + ap.k2 * r4 + ap.k3 * r6

        dx = xq * radial_factor + ap.p1 * (r2 + 2 * xq * xq) + 2 * ap.p2 * xq * yq
        dy = yq * radial_factor + ap.p2 * (r2 + 2 * yq * yq) + 2 * ap.p1 * xq * yq

        xq_new = (x + y * sin_she) * inv_scx - dx
        yq_new = y / cos_she - dy

        dx_change = xq_new - xq
        dy_change = yq_new - yq

        xq += DAMPING * dx_change
        yq += DAMPING * dy_change

        if np.sqrt(dx_change * dx_change + dy_change * dy_change) < tol:
            break

    return xq, yq

def flat_to_dist(flat_x, flat_y, cal):
    flat_x += cal.int_par.xh
    flat_y += cal.int_par.yh

    return distort_brown_affin(flat_x, flat_y, cal.added_par)

def dist_to_flat(dist_x, dist_y, cal, tol):
    flat_x, flat_y = correct_brown_affine_exact(dist_x, dist_y, cal.added_par, tol)
    flat_x -= cal.int_par.xh
    flat_y -= cal.int_par.yh

    return flat_x, flat_y
