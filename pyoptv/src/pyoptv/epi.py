import numpy as np
from typing import List, Tuple, Any
from .trafo import pixel_to_metric, dist_to_flat, metric_to_pixel, correct_brown_affin
from .imgcoord import flat_image_coord as imgcoord_flat_image_coord
from .ray_tracing import ray_tracing as real_ray_tracing
from .parameters import ControlPar, VolumePar, MMNP
from .calibration import Calibration
from .vec_utils import Vec2D, Vec3D

MAXCAND = 100  # Avoid circular import, match value from correspondences.py


def epi_mm(
    xl: float,
    yl: float,
    cal1: Calibration,
    cal2: Calibration,
    mmp: MMNP,
    vpar: VolumePar,
) -> Tuple[float, float, float, float]:
    pos, v = ray_tracing(xl, yl, cal1, mmp)
    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmin_lay[1] - vpar.Zmin_lay[0]) / (
        vpar.X_lay[1] - vpar.X_lay[0]
    )
    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmax_lay[1] - vpar.Zmax_lay[0]) / (
        vpar.X_lay[1] - vpar.X_lay[0]
    )
    xmin, ymin = flat_image_coord(move_along_ray(Zmin, pos, v), cal2, mmp)
    xmax, ymax = flat_image_coord(move_along_ray(Zmax, pos, v), cal2, mmp)
    return xmin, ymin, xmax, ymax


def epi_mm_2D(
    xl: float, yl: float, cal1: Calibration, mmp: MMNP, vpar: VolumePar
) -> np.ndarray:
    pos, v = ray_tracing(xl, yl, cal1, mmp)
    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmin_lay[1] - vpar.Zmin_lay[0]) / (
        vpar.X_lay[1] - vpar.X_lay[0]
    )
    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmax_lay[1] - vpar.Zmax_lay[0]) / (
        vpar.X_lay[1] - vpar.X_lay[0]
    )
    return move_along_ray(0.5 * (Zmin + Zmax), pos, v)


def find_candidate(
    crd: List[Vec2D],
    pix: List[Any],
    num: int,
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    n: int,
    nx: int,
    ny: int,
    sumg: float,
    vpar: VolumePar,
    cpar: ControlPar,
    cal: Calibration,
) -> Tuple[int, List[Tuple[int, float, float]]]:
    tol_band_width = vpar.eps0
    xmin = -cpar.pix_x * cpar.imx / 2 - cal.int_par.xh
    ymin = -cpar.pix_y * cpar.imy / 2 - cal.int_par.yh
    xmax = cpar.pix_x * cpar.imx / 2 - cal.int_par.xh
    ymax = cpar.pix_y * cpar.imy / 2 - cal.int_par.yh
    xmin, ymin = correct_brown_affin(xmin, ymin, cal.added_par)
    xmax, ymax = correct_brown_affin(xmax, ymax, cal.added_par)

    if xa == xb:
        xb += 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    if xa > xb:
        xa, xb = xb, xa
    if ya > yb:
        ya, yb = yb, ya

    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        return -1, []

    j0 = num // 2
    dj = num // 4
    while dj > 1:
        if crd[j0].x < (xa - tol_band_width):
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    candidates: List[Tuple[int, float, float]] = []
    for j in range(j0, num):
        if crd[j].x > xb + tol_band_width:
            return len(candidates), candidates

        if crd[j].y <= ya - tol_band_width or crd[j].y >= yb + tol_band_width:
            continue
        if crd[j].x <= xa - tol_band_width or crd[j].x >= xb + tol_band_width:
            continue

        d = abs((crd[j].y - m * crd[j].x - b) / np.sqrt(m * m + 1))
        if d >= tol_band_width:
            continue

        p2 = crd[j].pnr
        if p2 >= num:
            print("pnr out of range:", p2)
            return -1, []

        qn = quality_ratio(n, pix[p2].n)
        qnx = quality_ratio(nx, pix[p2].nx)
        qny = quality_ratio(ny, pix[p2].ny)
        qsumg = quality_ratio(sumg, pix[p2].sumg)

        if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
            continue
        if len(candidates) >= MAXCAND:
            print("More candidates than (maxcand):", len(candidates))
            return len(candidates), candidates

        corr = (4 * qsumg + 2 * qn + qnx + qny) * (sumg + pix[p2].sumg)
        candidates.append((j, d, corr))

    return len(candidates), candidates


def quality_ratio(a: float, b: float) -> float:
    return min(a, b) / max(a, b)


def move_along_ray(Z: float, pos: np.ndarray, v: np.ndarray) -> np.ndarray:
    return pos + (Z - pos[2]) * v / v[2]


def epipolar_curve(
    image_point: Tuple[float, float],
    origin_cam: Calibration,
    project_cam: Calibration,
    num_points: int,
    cparam: ControlPar,
    vpar: VolumePar,
) -> np.ndarray:
    img_pt = pixel_to_metric(image_point[0], image_point[1], cparam)
    img_pt = dist_to_flat(img_pt[0], img_pt[1], origin_cam, 1e-5)
    X = np.zeros(3)
    a = np.zeros(3)
    real_ray_tracing(img_pt[0], img_pt[1], origin_cam, cparam.mm, X, a)
    line_points = np.empty((num_points, 2))
    Zs = np.linspace(vpar.Zmin_lay[0], vpar.Zmax_lay[0], num_points)
    for pt_ix, Z in enumerate(Zs):
        pos = X + (Z - X[2]) / a[2] * a  # move_along_ray in C is just X + t*a
        x, y = imgcoord_flat_image_coord(pos, project_cam, cparam.mm)
        x, y = metric_to_pixel(x, y, cparam)
        line_points[pt_ix, 0] = x
        line_points[pt_ix, 1] = y
    return line_points

# Replace stubs with real functions
ray_tracing = real_ray_tracing
flat_image_coord = imgcoord_flat_image_coord
# correct_brown_affin is imported from trafo
