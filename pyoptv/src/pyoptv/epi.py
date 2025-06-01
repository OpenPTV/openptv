# type: ignore
import numpy as np
from typing import List, Tuple, Any
from .trafo import pixel_to_metric, dist_to_flat, metric_to_pixel, correct_brown_affin
from .imgcoord import flat_image_coord
from .ray_tracing import ray_tracing
from .parameters import ControlPar, VolumePar, MMNP
from .calibration import Calibration
from .epi import epi_mm, epi_mm_2D, find_candidate, Coord2D, Candidate
from .tracking_frame_buf import Target
from dataclasses import dataclass
from .multimed import move_along_ray


MAXCAND = 100  # Avoid circular import, match value from correspondences.py


@dataclass
class Candidate:
    pnr: int = -1 
    tol: float = np.nan
    corr: float = np.nan

@dataclass
class Coord2D:
    pnr: int = -1
    x: float = np.nan
    y: float = np.nan


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
    crd: List[Coord2D],
    pix: List[Target],
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
) -> Tuple[int, List[Candidate]]:
    tol_band_width = vpar.eps0
    xmin = -cpar.pix_x * cpar.imx / 2 - cal.int_par.xh
    ymin = -cpar.pix_y * cpar.imy / 2 - cal.int_par.yh
    xmax = cpar.pix_x * cpar.imx / 2 - cal.int_par.xh
    ymax = cpar.pix_y * cpar.imy / 2 - cal.int_par.yh
    xmin, ymin = correct_brown_affin(xmin, ymin, cal.added_par)
    xmax, ymax = correct_brown_affin(xmax, ymax, cal.added_par)

    # Debug output for epipolar line and search window
    print(f"find_candidate: xa={xa:.4f}, ya={ya:.4f}, xb={xb:.4f}, yb={yb:.4f}")
    print(f"find_candidate: xmin={xmin:.4f}, xmax={xmax:.4f}, ymin={ymin:.4f}, ymax={ymax:.4f}")
    print(f"find_candidate: tol_band_width={tol_band_width}")
    print(f"find_candidate: num candidates in crd={num}")
    for idx in range(min(num, 5)):
        print(f"  crd[{idx}]: pnr={crd[idx].pnr}, x={crd[idx].x:.4f}, y={crd[idx].y:.4f}")

    if xa == xb:
        xb += 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    if xa > xb:
        xa, xb = xb, xa
    if ya > yb:
        ya, yb = yb, ya

    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        # out-of-bounds epipolar strip: return -1 for semantic match to C implementation
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

    candidates: List[Candidate] = []
    for j in range(j0, num):
        if crd[j].x > xb + tol_band_width:
            print(f"find_candidate: reached x>{xb + tol_band_width:.4f}, breaking candidate loop")
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
            print("find_candidate: pnr out of range:", p2)
            return 0, []

        qn = quality_ratio(n, pix[p2].n)
        qnx = quality_ratio(nx, pix[p2].nx)
        qny = quality_ratio(ny, pix[p2].ny)
        qsumg = quality_ratio(sumg, pix[p2].sumg)

        if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
            continue
        if len(candidates) >= MAXCAND:
            print("find_candidate: More candidates than (maxcand):", len(candidates))
            return len(candidates), candidates

        corr = (4 * qsumg + 2 * qn + qnx + qny) * (sumg + pix[p2].sumg)
        print(f"find_candidate: candidate j={j}, pnr={p2}, d={d:.6f}, corr={corr:.2f}")
        candidates.append(Candidate(j, d, corr))

    print(f"find_candidate: returning {len(candidates)} candidates")
    return len(candidates), candidates


def quality_ratio(a: float, b: float) -> float:
    return min(a, b) / max(a, b)


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
    # Perform ray tracing to get origin and direction vectors
    X, a = ray_tracing(img_pt[0], img_pt[1], origin_cam, cparam.mm)
    line_points = np.empty((num_points, 2))
    Zs = np.linspace(vpar.Zmin_lay[0], vpar.Zmax_lay[0], num_points)
    for pt_ix, Z in enumerate(Zs):
        pos = X + (Z - X[2]) / a[2] * a  # move_along_ray in C is just X + t*a
        x, y = flat_image_coord(pos, project_cam, cparam.mm)
        x, y = metric_to_pixel(x, y, cparam)
        line_points[pt_ix, 0] = x
        line_points[pt_ix, 1] = y
    return line_points
