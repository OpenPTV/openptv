import numpy as np
from typing import Tuple
from pyoptv.calibration import Calibration, Glass, Exterior
from pyoptv.parameters import ControlPar, MMNP, VolumePar

def get_mmf_from_mmlut(cal: Calibration, pos: np.ndarray) -> float:
    """
    Interpolates the multi-media factor (mmf) from the calibration's mmlut grid
    for a given 3D position.
    """
    rw = cal.mmlut.rw
    temp = pos - cal.mmlut.origin
    sz = temp[2] / rw
    iz = int(sz)
    sz -= iz
    R = np.linalg.norm(temp[:2])
    sr = R / rw
    ir = int(sr)
    sr -= ir
    nz = cal.mmlut.nz
    nr = cal.mmlut.nr
    if ir > nr or iz < 0 or iz > nz:
        return 0
    v4 = [
        ir * nz + iz,
        ir * nz + (iz + 1),
        (ir + 1) * nz + iz,
        (ir + 1) * nz + (iz + 1),
    ]
    for i in range(4):
        if v4[i] < 0 or v4[i] > nr * nz:
            return 0
    mmf = (
        cal.mmlut.data[v4[0]] * (1 - sr) * (1 - sz)
        + cal.mmlut.data[v4[1]] * (1 - sr) * sz
        + cal.mmlut.data[v4[2]] * sr * (1 - sz)
        + cal.mmlut.data[v4[3]] * sr * sz
    )
    return mmf

def multimed_nlay(cal: Calibration, mm: MMNP, pos: np.ndarray) -> Tuple[float, float]:
    """
    Applies multi-media correction for n-layer model.
    Returns the corrected X, Y coordinates.
    """
    radial_shift = multimed_r_nlay(cal, mm, pos)
    Xq = cal.ext_par.x0 + (pos[0] - cal.ext_par.x0) * radial_shift
    Yq = cal.ext_par.y0 + (pos[1] - cal.ext_par.y0) * radial_shift
    return Xq, Yq

def multimed_r_nlay(cal: Calibration, mm: MMNP, pos: np.ndarray) -> float:
    """
    Computes the radial shift for a point in a multi-media, n-layer model.
    Returns the scaling factor for the radial correction.
    """
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0
    if cal.mmlut.data is not None:
        mmf = get_mmf_from_mmlut(cal, pos)
        if mmf > 0:
            return mmf
    X, Y, Z = pos
    zout = Z + sum(mm.d[1 : mm.nlay])
    r = np.linalg.norm([X - cal.ext_par.x0, Y - cal.ext_par.y0])
    rq = r
    # Iteratively solve for the radial correction
    for _ in range(40):
        beta1 = np.arctan(rq / (cal.ext_par.z0 - Z))
        beta2 = [np.arcsin(np.sin(beta1) * mm.n1 / mm.n2[i]) for i in range(mm.nlay)]
        beta3 = np.arcsin(np.sin(beta1) * mm.n1 / mm.n3)
        rbeta = (
            (cal.ext_par.z0 - mm.d[0]) * np.tan(beta1)
            - zout * np.tan(beta3)
            + sum(mm.d[i] * np.tan(beta2[i]) for i in range(mm.nlay))
        )
        rdiff = r - rbeta
        rq += rdiff
        if abs(rdiff) < 0.001:
            break
    return rq / r if r != 0 else 1.0

def trans_Cam_Point(
    ex: Exterior, mm: MMNP, gl: Glass, pos: np.ndarray
) -> Tuple[Exterior, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms a point from camera coordinates to glass coordinates,
    considering the glass interface and multi-media parameters.
    Returns the transformed exterior, transformed position, and crossing points.
    """
    glass_dir = np.array([gl.vec_x, gl.vec_y, gl.vec_z])
    dist_o_glas = np.linalg.norm(glass_dir)
    if np.isclose(dist_o_glas, 0.0):
        # No glass, return original values
        ex_t = Exterior()
        ex_t.x0 = ex.x0
        ex_t.y0 = ex.y0
        ex_t.z0 = ex.z0
        return ex_t, pos, pos, pos
    primary_pt = np.array([ex.x0, ex.y0, ex.z0])
    dist_cam_glas = (
        np.dot(primary_pt, glass_dir) / dist_o_glas - dist_o_glas - mm.d[0]
    )
    dist_point_glas = np.dot(pos, glass_dir) / dist_o_glas - dist_o_glas
    renorm_glass = glass_dir * (dist_cam_glas / dist_o_glas)
    cross_c = primary_pt - renorm_glass
    renorm_glass = glass_dir * (dist_point_glas / dist_o_glas)
    cross_p = pos - renorm_glass
    ex_t = Exterior()
    ex_t.x0 = 0.0
    ex_t.y0 = 0.0
    ex_t.z0 = dist_cam_glas + mm.d[0]
    renorm_glass = glass_dir * (mm.d[0] / dist_o_glas)
    temp = cross_c - renorm_glass
    pos_t = cross_p - temp
    return ex_t, pos_t, cross_p, cross_c

def back_trans_Point(
    pos_t, mm: MMNP, G: Glass, cross_p, cross_c
) -> np.ndarray:
    """
    Back-transforms a point from glass coordinates to camera coordinates.
    Accepts pos_t as either a numpy array or a Vec3D.
    """
    glass_dir = np.array([G.vec_x, G.vec_y, G.vec_z])
    nGl = np.linalg.norm(glass_dir)
    if np.isclose(nGl, 0.0):
        # Avoid division by zero if glass vector is zero (no glass)
        if hasattr(pos_t, 'x') and hasattr(pos_t, 'y') and hasattr(pos_t, 'z'):
            return np.array([pos_t.x, pos_t.y, pos_t.z])
        else:
            return np.array([pos_t[0], pos_t[1], pos_t[2]])
    renorm_glass = glass_dir * (mm.d[0] / nGl)
    after_glass = cross_c - renorm_glass
    temp = cross_p - after_glass
    nVe = np.linalg.norm(temp)
    # Support both numpy array and Vec3D for pos_t
    if hasattr(pos_t, 'x') and hasattr(pos_t, 'y') and hasattr(pos_t, 'z'):
        pt0, pt2 = pos_t.x, pos_t.z
    else:
        pt0, pt2 = pos_t[0], pos_t[2]
    renorm_glass = glass_dir * (-pt2 / nGl)
    pos = after_glass - renorm_glass
    if nVe > 0:
        renorm_glass = temp * (-pt0 / nVe)
        pos -= renorm_glass
    return pos

def move_along_ray(glob_Z: float, vertex: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """
    Moves a point along a ray to a specified Z coordinate.
    """
    out = vertex + (glob_Z - vertex[2]) * direct / direct[2]
    return out

def init_mmlut(vpar: VolumePar, cpar: ControlPar, cal: Calibration) -> None:
    """
    Initializes the multi-media lookup table (mmlut) for a given calibration.
    """
    rw = 2.0
    cal_t = Calibration()
    cal_t.ext_par = cal.ext_par
    cal_t.int_par = cal.int_par
    cal_t.glass_par = cal.glass_par
    cal_t.added_par = cal.added_par
    cal_t.mmlut = cal.mmlut
    Zmin = min(vpar.Zmin_lay)
    Zmax = max(vpar.Zmax_lay)
    Zmin -= Zmin % rw
    Zmax += rw - Zmax % rw
    Zmin_t = Zmin
    Zmax_t = Zmax
    Rmax = 0
    # Compute the bounding box in Z and R
    for x, y in [
        (0, 0),
        (cpar.imx, 0),
        (0, cpar.imy),
        (cpar.imx, cpar.imy),
    ]:
        x -= cal.int_par.xh
        y -= cal.int_par.yh
        x, y = correct_brown_affin(x, y, cal.added_par)
        pos, a = ray_tracing(x, y, cal, cpar.mm)
        xyz = move_along_ray(Zmin, pos, a)
        ex_t, xyz_t, cross_p, cross_c = trans_Cam_Point(
            cal.ext_par, cpar.mm, cal.glass_par, xyz
        )
        Zmin_t = min(Zmin_t, xyz_t[2])
        Zmax_t = max(Zmax_t, xyz_t[2])
        Rmax = max(Rmax, np.linalg.norm([xyz_t[0] - ex_t.x0, xyz_t[1] - ex_t.y0]))
        xyz = move_along_ray(Zmax, pos, a)
        ex_t, xyz_t, cross_p, cross_c = trans_Cam_Point(
            cal.ext_par, cpar.mm, cal.glass_par, xyz
        )
        Zmin_t = min(Zmin_t, xyz_t[2])
        Zmax_t = max(Zmax_t, xyz_t[2])
        Rmax = max(Rmax, np.linalg.norm([xyz_t[0] - ex_t.x0, xyz_t[1] - ex_t.y0]))
    Rmax += rw - Rmax % rw
    nr = int(Rmax / rw + 1)
    nz = int((Zmax_t - Zmin_t) / rw + 1)
    cal.mmlut.origin = np.array([cal_t.ext_par.x0, cal_t.ext_par.y0, Zmin_t])
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = rw
    data = np.zeros((nr, nz))
    # Fill the lookup table with radial correction factors
    for i in range(nr):
        for j in range(nz):
            xyz = np.array([i * rw + cal_t.ext_par.x0, cal_t.ext_par.y0, Zmin_t + j * rw])
            data[i, j] = multimed_r_nlay(cal_t, cpar.mm, xyz)
    cal.mmlut.data = data

def volumedimension(
    xmax: float,
    xmin: float,
    ymax: float,
    ymin: float,
    zmax: float,
    zmin: float,
    vpar: VolumePar,
    cpar: ControlPar,
    cal: list,
) -> None:
    """
    Computes the bounding box of the measurement volume in 3D space.
    Updates xmax, xmin, ymax, ymin, zmax, zmin in place.
    """
    Zmin = min(vpar.Zmin_lay)
    Zmax = max(vpar.Zmax_lay)
    zmin, zmax = Zmin, Zmax
    for i_cam in range(cpar.num_cams):
        for x, y in [
            (0, 0),
            (cpar.imx, 0),
            (0, cpar.imy),
            (cpar.imx, cpar.imy),
        ]:
            x -= cal[i_cam].int_par.xh
            y -= cal[i_cam].int_par.yh
            x, y = correct_brown_affin(x, y, cal[i_cam].added_par)
            pos, a = ray_tracing(x, y, cal[i_cam], cpar.mm)
            X = pos[0] + (Zmin - pos[2]) * a[0] / a[2]
            Y = pos[1] + (Zmin - pos[2]) * a[1] / a[2]
            xmax, xmin = max(xmax, X), min(xmin, X)
            ymax, ymin = max(ymax, Y), min(ymin, Y)
            X = pos[0] + (Zmax - pos[2]) * a[0] / a[2]
            Y = pos[1] + (Zmax - pos[2]) * a[1] / a[2]
            xmax, xmin = max(xmax, X), min(xmin, X)
            ymax, ymin = max(ymax, Y), min(ymin, Y)
