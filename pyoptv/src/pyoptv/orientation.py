# type: ignore
import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pyoptv.calibration import Calibration, Exterior
from pyoptv.parameters import ControlPar
from pyoptv.trafo import pixel_to_metric, correct_brown_affin
from pyoptv.imgcoord import img_coord
from pyoptv.ray_tracing import ray_tracing
from pyoptv.sortgrid import read_calblock

COORD_UNUSED = -1e10
IDT = 10
NPAR = 19
NUM_ITER = 80
POS_INF = 1E20
CONVERGENCE = 0.00001

class OrientPar:
    """
    Parameter flags for orientation adjustment.
    Each flag controls whether a parameter is included in the adjustment.
    """
    def __init__(
        self,
        useflag: int = 0,
        ccflag: int = 0,
        xhflag: int = 0,
        yhflag: int = 0,
        k1flag: int = 0,
        k2flag: int = 0,
        k3flag: int = 0,
        p1flag: int = 0,
        p2flag: int = 0,
        scxflag: int = 0,
        sheflag: int = 0,
        interfflag: int = 0,
    ):
        self.useflag = useflag
        self.ccflag = ccflag
        self.xhflag = xhflag
        self.yhflag = yhflag
        self.k1flag = k1flag
        self.k2flag = k2flag
        self.k3flag = k3flag
        self.p1flag = p1flag
        self.p2flag = p2flag
        self.scxflag = scxflag
        self.sheflag = sheflag
        self.interfflag = interfflag

def skew_midpoint(
    vert1: np.ndarray, direct1: np.ndarray, vert2: np.ndarray, direct2: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Computes the shortest distance and midpoint between two skew lines in 3D.
    """
    sp_diff = vert2 - vert1
    perp_both = np.cross(direct1, direct2)
    scale = np.dot(perp_both, perp_both)
    temp = np.cross(sp_diff, direct2)
    on1 = vert1 + np.dot(perp_both, temp) / scale * direct1
    temp = np.cross(sp_diff, direct1)
    on2 = vert2 + np.dot(perp_both, temp) / scale * direct2
    res = (on1 + on2) / 2
    return np.linalg.norm(on1 - on2), res

def point_position(
    targets: np.ndarray,
    num_cams: int,
    multimed_pars,
    cals: List[Calibration],
) -> Tuple[float, np.ndarray]:
    """
    Computes the average intersection point and mean distance between rays from multiple cameras.
    """
    num_used_pairs = 0
    dtot = 0.0
    point_tot = np.zeros(3)
    vertices = np.zeros((num_cams, 3))
    directs = np.zeros((num_cams, 3))
    for cam in range(num_cams):
        if targets[cam, 0] != COORD_UNUSED:
            vertices[cam], directs[cam] = ray_tracing(
                targets[cam, 0], targets[cam, 1], cals[cam], multimed_pars
            )
    for cam in range(num_cams):
        if targets[cam, 0] == COORD_UNUSED:
            continue
        for pair in range(cam + 1, num_cams):
            if targets[pair, 0] == COORD_UNUSED:
                continue
            num_used_pairs += 1
            dist, point = skew_midpoint(
                vertices[cam], directs[cam], vertices[pair], directs[pair]
            )
            dtot += dist
            point_tot += point
    res = point_tot / num_used_pairs
    return dtot / num_used_pairs, res

def weighted_dumbbell_precision(
    targets: np.ndarray,
    num_targs: int,
    num_cams: int,
    multimed_pars,
    cals: List[Calibration],
    db_length: float,
    db_weight: float,
) -> float:
    """
    Computes a weighted precision metric for a dumbbell calibration object.
    """
    dtot = 0.0
    len_err_tot = 0.0
    res = np.zeros((2, 3))
    for pt in range(num_targs):
        res_current = res[pt % 2]
        dist, res_current = point_position(targets[pt], num_cams, multimed_pars, cals)
        dtot += dist
        if pt % 2 == 1:
            dist = np.linalg.norm(res[0] - res[1])
            len_err_tot += 1 - (db_length / dist if dist > db_length else dist / db_length)
    return dtot / num_targs + db_weight * len_err_tot / (0.5 * num_targs)

def num_deriv_exterior(
    cal: Calibration,
    cpar: ControlPar,
    dpos: float,
    dang: float,
    pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically computes derivatives of image coordinates with respect to exterior orientation parameters.
    NOTE: This function mutates cal.ext_par fields in-place for finite differencing.
    """
    x_ders = np.zeros(6)
    y_ders = np.zeros(6)
    vars = [
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        cal.ext_par.omega,
        cal.ext_par.phi,
        cal.ext_par.kappa,
    ]
    xs, ys = img_coord(pos, cal, cpar.mm)
    for pd in range(6):
        step = dang if pd > 2 else dpos
        vars[pd] += step
        if pd > 2:
            cal.ext_par.dm = Calibration.rotation_matrix(
                cal.ext_par.omega, cal.ext_par.phi, cal.ext_par.kappa
            )
        xpd, ypd = img_coord(pos, cal, cpar.mm)
        x_ders[pd] = (xpd - xs) / step
        y_ders[pd] = (ypd - ys) / step
        vars[pd] -= step
    cal.ext_par.dm = Calibration.rotation_matrix(
        cal.ext_par.omega, cal.ext_par.phi, cal.ext_par.kappa
    )
    return x_ders, y_ders

def orient(
    cal_in: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    pix: List,
    flags: OrientPar,
    sigmabeta: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Performs iterative orientation adjustment for camera calibration.
    Updates cal_in in-place if converged.
    Returns residuals if converged, else None.
    NOTE: This function mutates cal_in and expects pix to have .pnr attribute.
    """
    cal = cal_in.copy()
    maxsize = nfix * 2 + IDT
    P = np.ones(maxsize)
    y = np.zeros(maxsize)
    X = np.zeros((maxsize, NPAR))
    ident = [cal.int_par.cc, cal.int_par.xh, cal.int_par.yh, cal.added_par.k1, cal.added_par.k2, cal.added_par.k3, cal.added_par.p1, cal.added_par.p2, cal.added_par.scx, cal.added_par.she]
    safety_x, safety_y, safety_z = cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z
    glass_dir = np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
    nGl = np.linalg.norm(glass_dir)
    e1 = np.array([2 * cal.glass_par.vec_z - 3 * cal.glass_par.vec_x, 3 * cal.glass_par.vec_x - 1 * cal.glass_par.vec_z, 1 * cal.glass_par.vec_y - 2 * cal.glass_par.vec_y])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, glass_dir)
    al, be, ga = 0, 0, 0
    itnum = 0
    stopflag = 0
    while stopflag == 0 and itnum < NUM_ITER:
        itnum += 1
        n = 0
        for i in range(nfix):
            if pix[i].pnr != i:
                continue
            if flags.useflag == 1 and i % 2 == 0:
                continue
            if flags.useflag == 2 and i % 2 != 0:
                continue
            if flags.useflag == 3 and i % 3 == 0:
                continue
            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)
            xc, yc = correct_brown_affin(xc, yc, cal.added_par)
            xp, yp = img_coord(fix[i], cal, cpar.mm)
            r = np.sqrt(xp ** 2 + yp ** 2)
            X[n, 7] = cal.added_par.scx
            X[n + 1, 7] = np.sin(cal.added_par.she)
            X[n, 8] = 0
            X[n + 1, 8] = 1
            X[n, 9] = cal.added_par.scx * xp * r ** 2
            X[n + 1, 9] = yp * r ** 2
            X[n, 10] = cal.added_par.scx * xp * r ** 4
            X[n + 1, 10] = yp * r ** 4
            X[n, 11] = cal.added_par.scx * xp * r ** 6
            X[n + 1, 11] = yp * r ** 6
            X[n, 12] = cal.added_par.scx * (2 * xp ** 2 + r ** 2)
            X[n + 1, 12] = 2 * xp * yp
            X[n, 13] = 2 * cal.added_par.scx * xp * yp
            X[n + 1, 13] = 2 * yp ** 2 + r ** 2
            qq = cal.added_par.k1 * r ** 2 + cal.added_par.k2 * r ** 4 + cal.added_par.k3 * r ** 6 + 1
            X[n, 14] = xp * qq + cal.added_par.p1 * (r ** 2 + 2 * xp ** 2) + 2 * cal.added_par.p2 * xp * yp
            X[n + 1, 14] = 0
            X[n, 15] = -np.cos(cal.added_par.she) * yp
            X[n + 1, 15] = -np.sin(cal.added_par.she) * yp
            x_ders, y_ders = num_deriv_exterior(cal, cpar, 0.00001, 0.0000001, fix[i])
            X[n, :6] = x_ders
            X[n + 1, :6] = y_ders
            cal.int_par.cc += 0.00001
            xp_d, yp_d = img_coord(fix[i], cal, cpar.mm)
            X[n, 6] = (xp_d - xp) / 0.00001
            X[n + 1, 6] = (yp_d - yp) / 0.00001
            cal.int_par.cc -= 0.00001
            al += 0.00001
            cal.glass_par.vec_x += e1[0] * nGl * al
            cal.glass_par.vec_y += e1[1] * nGl * al
            cal.glass_par.vec_z += e1[2] * nGl * al
            xp_d, yp_d = img_coord(fix[i], cal, cpar.mm)
            X[n, 16] = (xp_d - xp) / 0.00001
            X[n + 1, 16] = (yp_d - yp) / 0.00001
            al -= 0.00001
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z
            be += 0.00001
            cal.glass_par.vec_x += e2[0] * nGl * be
            cal.glass_par.vec_y += e2[1] * nGl * be
            cal.glass_par.vec_z += e2[2] * nGl * be
            xp_d, yp_d = img_coord(fix[i], cal, cpar.mm)
            X[n, 17] = (xp_d - xp) / 0.00001
            X[n + 1, 17] = (yp_d - yp) / 0.00001
            be -= 0.00001
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z
            ga += 0.00001
            cal.glass_par.vec_x += cal.glass_par.vec_x * nGl * ga
            cal.glass_par.vec_y += cal.glass_par.vec_y * nGl * ga
            cal.glass_par.vec_z += cal.glass_par.vec_z * nGl * ga
            xp_d, yp_d = img_coord(fix[i], cal, cpar.mm)
            X[n, 18] = (xp_d - xp) / 0.00001
            X[n + 1, 18] = (yp_d - yp) / 0.00001
            ga -= 0.00001
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z
            y[n] = xc - xp
            y[n + 1] = yc - yp
            n += 2
        n_obs = n
        for i in range(IDT):
            X[n_obs + i, 6 + i] = 1
        y[n_obs:n_obs + IDT] = ident - np.array([cal.int_par.cc, cal.int_par.xh, cal.int_par.yh, cal.added_par.k1, cal.added_par.k2, cal.added_par.k3, cal.added_par.p1, cal.added_par.p2, cal.added_par.scx, cal.added_par.she])
        P[n_obs:n_obs + IDT] = [POS_INF if not flag else 1 for flag in [flags.ccflag, flags.xhflag, flags.yhflag, flags.k1flag, flags.k2flag, flags.k3flag, flags.p1flag, flags.p2flag, flags.scxflag, flags.sheflag]]
        n_obs += IDT
        sumP = np.sum(P[:n_obs])
        p = np.sqrt(P[:n_obs])
        Xh = X[:n_obs] * p[:, np.newaxis]
        yh = y[:n_obs] * p
        XPX = np.linalg.inv(Xh.T @ Xh)
        XPy = Xh.T @ yh
        beta = XPX @ XPy
        stopflag = np.all(np.abs(beta[:16]) <= CONVERGENCE)
        cal.ext_par.x0 += beta[0]
        cal.ext_par.y0 += beta[1]
        cal.ext_par.z0 += beta[2]
        cal.ext_par.omega += beta[3]
        cal.ext_par.phi += beta[4]
        cal.ext_par.kappa += beta[5]
        cal.int_par.cc += beta[6]
        cal.int_par.xh += beta[7]
        cal.int_par.yh += beta[8]
        cal.added_par.k1 += beta[9]
        cal.added_par.k2 += beta[10]
        cal.added_par.k3 += beta[11]
        cal.added_par.p1 += beta[12]
        cal.added_par.p2 += beta[13]
        cal.added_par.scx += beta[14]
        cal.added_par.she += beta[15]
        if flags.interfflag:
            cal.glass_par.vec_x += e1[0] * nGl * beta[16]
            cal.glass_par.vec_y += e1[1] * nGl * beta[16]
            cal.glass_par.vec_z += e1[2] * nGl * beta[16]
            cal.glass_par.vec_x += e2[0] * nGl * beta[17]
            cal.glass_par.vec_y += e2[1] * nGl * beta[17]
            cal.glass_par.vec_z += e2[2] * nGl * beta[17]
    resi = X @ beta - y
    omega = np.sum(resi[:n_obs] ** 2 * P[:n_obs])
    sigmabeta[:NPAR] = np.sqrt(np.diag(XPX) * omega / (n_obs - 16))
    sigmabeta[NPAR] = np.sqrt(omega / (n_obs - 16))
    if stopflag:
        cal_in.update(cal)
        return resi
    else:
        return None

def raw_orient(
    cal: Calibration,
    cpar: ControlPar,
    nfix: int,
    fix: np.ndarray,
    pix: List,
) -> bool:
    """
    Performs a raw orientation adjustment (no distortion parameters).
    Returns True if converged, else False.
    """
    X = np.zeros((10, 6))
    y = np.zeros(10)
    cal.added_par.k1 = 0
    cal.added_par.k2 = 0
    cal.added_par.k3 = 0
    cal.added_par.p1 = 0
    cal.added_par.p2 = 0
    cal.added_par.scx = 1
    cal.added_par.she = 0
    itnum = 0
    stopflag = 0
    while stopflag == 0 and itnum < 20:
        itnum += 1
        n = 0
        for i in range(nfix):
            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)
            xp, yp = img_coord(fix[i], cal, cpar.mm)
            x_ders, y_ders = num_deriv_exterior(cal, cpar, 0.0001, 0.0001, fix[i])
            X[n, :6] = x_ders
            X[n + 1, :6] = y_ders
            y[n] = xc - xp
            y[n + 1] = yc - yp
            n += 2
        XPX = np.linalg.inv(X[:n].T @ X[:n])
        XPy = X[:n].T @ y[:n]
        beta = XPX @ XPy
        stopflag = np.all(np.abs(beta) <= 0.1)
        cal.ext_par.x0 += beta[0]
        cal.ext_par.y0 += beta[1]
        cal.ext_par.z0 += beta[2]
        cal.ext_par.omega += beta[3]
        cal.ext_par.phi += beta[4]
        cal.ext_par.kappa += beta[5]
    if stopflag:
        rotation_matrix(cal.ext_par)
    return stopflag

def read_man_ori_fix(
    fix4: np.ndarray,
    calblock_filename: str,
    man_ori_filename: str,
    cam: int,
) -> int:
    """
    Reads four manually oriented calibration points from file and fills fix4.
    Returns the number of matches found.
    """
    with open(man_ori_filename, "r") as fpp:
        for _ in range(cam):
            fpp.readline()
        nr = list(map(int, fpp.readline().split()))
    fix = read_calblock(calblock_filename)
    num_match = 0
    for pnr, point in enumerate(fix):
        if pnr in nr:
            fix4[nr.index(pnr)] = point
            num_match += 1
        if num_match >= 4:
            break
    return num_match

def read_orient_par(filename: str) -> OrientPar:
    """
    Reads orientation parameter flags from a file and returns an OrientPar instance.
    """
    with open(filename, "r") as file:
        params = list(map(int, file.read().split()))
    return OrientPar(*params)
