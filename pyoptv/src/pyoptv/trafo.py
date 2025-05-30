import numpy as np
from typing import Tuple, Any
from .parameters import ControlPar
from .calibration import ap_52, Calibration


MAX_ITER = 50
DAMPING = 0.5
TOL = 1e-8


def old_pixel_to_metric(
    x_pixel: float,
    y_pixel: float,
    im_size_x: float,
    im_size_y: float,
    pix_size_x: float,
    pix_size_y: float,
    y_remap_mode: int,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to metric coordinates using legacy formula.
    """
    if y_remap_mode == 1:
        y_pixel = 2.0 * y_pixel + 1.0
    elif y_remap_mode == 2:
        y_pixel *= 2.0

    x_metric = (x_pixel - im_size_x / 2.0) * pix_size_x
    y_metric = (im_size_y / 2.0 - y_pixel) * pix_size_y

    return x_metric, y_metric


def pixel_to_metric(
    x_pixel: float, y_pixel: float, cpar: ControlPar
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to metric coordinates using camera parameters.
    """
    return old_pixel_to_metric(
        x_pixel,
        y_pixel,
        cpar.imx,
        cpar.imy,
        cpar.pix_x,
        cpar.pix_y,
        cpar.chfield,
    )


def old_metric_to_pixel(
    x_metric: float,
    y_metric: float,
    im_size_x: float,
    im_size_y: float,
    pix_size_x: float,
    pix_size_y: float,
    y_remap_mode: int,
) -> Tuple[float, float]:
    """
    Convert metric coordinates to pixel coordinates using legacy formula.
    """
    x_pixel = (x_metric / pix_size_x) + (im_size_x / 2.0)
    y_pixel = (im_size_y / 2.0) - (y_metric / pix_size_y)

    if y_remap_mode == 1:
        y_pixel = (y_pixel - 1.0) / 2.0
    elif y_remap_mode == 2:
        y_pixel /= 2.0

    return x_pixel, y_pixel


def metric_to_pixel(
    x_metric: float, y_metric: float, cpar: ControlPar
) -> Tuple[float, float]:
    """
    Convert metric coordinates to pixel coordinates using camera parameters.
    """
    return old_metric_to_pixel(
        x_metric,
        y_metric,
        cpar.imx,
        cpar.imy,
        cpar.pix_x,
        cpar.pix_y,
        cpar.chfield,
    )


def distort_brown_affin(x: float, y: float, ap: ap_52) -> Tuple[float, float]:
    """
    Apply Brown distortion and affine transformation to coordinates.
    Args:
        x: x coordinate in flat (undistorted) space
        y: y coordinate in flat (undistorted) space
        ap: ap_52 object containing distortion parameters
    Returns:
        Tuple[float, float]: distorted x and y coordinates
    """
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


def correct_brown_affin(x: float, y: float, ap: ap_52) -> Tuple[float, float]:
    """
    Corrects the distortion using the Brown model with affine transformation.
    Args:
        x: x coordinate in distorted space
        y: y coordinate in distorted space
        ap: ap_52 object containing the distortion parameters
    Returns:
        Tuple[float, float]: corrected x and y coordinates in flat (undistorted) space
    """


    sin_she = np.sin(ap.she)
    cos_she = np.cos(ap.she)
    inv_scx = 1.0 / ap.scx

    xq = x * inv_scx
    yq = y * inv_scx / cos_she
    xq += yq * sin_she



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


def correct_brown_affine_exact(
    x: float, y: float, ap: ap_52, tol: float
) -> Tuple[float, float]:
    """
    Iteratively corrects Brown distortion and affine transformation to a given tolerance.
    Args:
        x: x coordinate in distorted space
        y: y coordinate in distorted space
        ap: ap_52 object containing the distortion parameters
        tol: tolerance for convergence
    Returns:
        Tuple[float, float]: corrected x and y coordinates in flat (undistorted) space
    """
    r_init = np.sqrt(x * x + y * y)
    if r_init < 1e-10:
        return 0.0, 0.0

    sin_she = np.sin(ap.she)
    cos_she = np.cos(ap.she)
    inv_scx = 1.0 / ap.scx

    xq = (x + y * sin_she) * inv_scx
    yq = y / cos_she


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


def flat_to_dist(flat_x: float, flat_y: float, cal: Calibration) -> Tuple[float, float]:
    """
    Convert flat (undistorted) coordinates to distorted coordinates using calibration.
    Args:
        flat_x: x coordinate in flat space
        flat_y: y coordinate in flat space
        cal: Calibration object
    Returns:
        Tuple[float, float]: distorted x and y coordinates
    """
    flat_x += cal.int_par.xh
    flat_y += cal.int_par.yh

    return distort_brown_affin(flat_x, flat_y, cal.added_par)


def dist_to_flat(
    dist_x: float, dist_y: float, cal: Calibration, tol: float
) -> Tuple[float, float]:
    """
    Convert distorted coordinates to flat (undistorted) coordinates using calibration.
    Args:
        dist_x: x coordinate in distorted space
        dist_y: y coordinate in distorted space
        cal: Calibration object
        tol: tolerance for convergence
    Returns:
        Tuple[float, float]: flat (undistorted) x and y coordinates
    """
    flat_x, flat_y = correct_brown_affine_exact(dist_x, dist_y, cal.added_par, tol)
    flat_x -= cal.int_par.xh
    flat_y -= cal.int_par.yh

    return flat_x, flat_y
