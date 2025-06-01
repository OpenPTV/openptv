# type: ignore
import numpy as np
from typing import List

SEQ_FNAME_MAX_LEN: int = 240

class SequencePar:
    num_cams: int
    img_base_name: List[str]
    first: int
    last: int
    def __init__(self, num_cams: int) -> None:
        self.num_cams = num_cams
        self.img_base_name = [""] * num_cams
        self.first = 0
        self.last = 0

def read_sequence_par(filename, num_cams):
    ret = SequencePar(num_cams)
    with open(filename, "r") as par_file:
        for cam in range(num_cams):
            ret.img_base_name[cam] = par_file.readline().strip()
        ret.first = int(par_file.readline().strip())
        ret.last = int(par_file.readline().strip())
    return ret

def new_sequence_par(num_cams):
    return SequencePar(num_cams)

def free_sequence_par(sp):
    del sp

def compare_sequence_par(sp1, sp2):
    if sp1.first != sp2.first or sp1.last != sp2.last or sp1.num_cams != sp2.num_cams:
        return False
    for cam in range(sp1.num_cams):
        if sp1.img_base_name[cam] != sp2.img_base_name[cam]:
            return False
    return True

class TrackPar:
    dacc: float
    dangle: float
    dvxmax: float
    dvxmin: float
    dvymax: float
    dvymin: float
    dvzmax: float
    dvzmin: float
    dsumg: int
    dn: int
    dnx: int
    dny: int
    add: int
    def __init__(self) -> None:
        self.dacc = 0.0
        self.dangle = 0.0
        self.dvxmax = 0.0
        self.dvxmin = 0.0
        self.dvymax = 0.0
        self.dvymin = 0.0
        self.dvzmax = 0.0
        self.dvzmin = 0.0
        self.dsumg = 0
        self.dn = 0
        self.dnx = 0
        self.dny = 0
        self.add = 0

def read_track_par(filename):
    ret = TrackPar()
    with open(filename, "r") as fpp:
        ret.dvxmin = float(fpp.readline().strip())
        ret.dvxmax = float(fpp.readline().strip())
        ret.dvymin = float(fpp.readline().strip())
        ret.dvymax = float(fpp.readline().strip())
        ret.dvzmin = float(fpp.readline().strip())
        ret.dvzmax = float(fpp.readline().strip())
        ret.dangle = float(fpp.readline().strip())
        ret.dacc = float(fpp.readline().strip())
        ret.add = int(fpp.readline().strip())
    return ret

def compare_track_par(t1, t2):
    return (t1.dvxmin == t2.dvxmin and t1.dvxmax == t2.dvxmax and
            t1.dvymin == t2.dvymin and t1.dvymax == t2.dvymax and
            t1.dvzmin == t2.dvzmin and t1.dvzmax == t2.dvzmax and
            t1.dacc == t2.dacc and t1.dangle == t2.dangle and
            t1.dsumg == t2.dsumg and t1.dn == t2.dn and
            t1.dnx == t2.dnx and t1.dny == t2.dny and t1.add == t2.add)

class VolumePar:
    def __init__(
        self,
        X_lay: List[float] = None,
        Zmin_lay: List[float] = None,
        Zmax_lay: List[float] = None,
        cnx: float = 0.3,
        cny: float = 0.3,
        cn: float = 0.01,
        csumg: float = 0.01,
        corrmin: float = 33.0,
        eps0: float = 1.0,
    ) -> None:
        self.X_lay = X_lay if X_lay is not None else [-100.0, 100.0]
        self.Zmin_lay = Zmin_lay if Zmin_lay is not None else [-100.0, -100.0]
        self.Zmax_lay = Zmax_lay if Zmax_lay is not None else [100.0, 100.0]
        self.cnx = cnx
        self.cny = cny
        self.cn = cn
        self.csumg = csumg
        self.corrmin = corrmin
        self.eps0 = eps0

def read_volume_par(filename):
    ret = VolumePar()
    with open(filename, "r") as fpp:
        ret.X_lay[0] = float(fpp.readline().strip())
        ret.Zmin_lay[0] = float(fpp.readline().strip())
        ret.Zmax_lay[0] = float(fpp.readline().strip())
        ret.X_lay[1] = float(fpp.readline().strip())
        ret.Zmin_lay[1] = float(fpp.readline().strip())
        ret.Zmax_lay[1] = float(fpp.readline().strip())
        ret.cnx = float(fpp.readline().strip())
        ret.cny = float(fpp.readline().strip())
        ret.cn = float(fpp.readline().strip())
        ret.csumg = float(fpp.readline().strip())
        ret.corrmin = float(fpp.readline().strip())
        ret.eps0 = float(fpp.readline().strip())
    return ret

def compare_volume_par(v1, v2):
    return (
        v1.X_lay == v2.X_lay and
        v1.Zmin_lay == v2.Zmin_lay and
        v1.Zmax_lay == v2.Zmax_lay and
        v1.cnx == v2.cnx and
        v1.cny == v2.cny and
        v1.cn == v2.cn and
        v1.csumg == v2.csumg and
        v1.corrmin == v2.corrmin and
        v1.eps0 == v2.eps0
    )

class MMNP:
    nlay: int
    n1: float
    d: List[float]
    n2: List[float]
    n3: float

    def __init__(
        self,
        nlay: int = 1,
        n1: float = 1.0,
        d: List[float] = None,
        n2: List[float] = None,
        n3: float = 1.0,
    ) -> None:
        self.nlay = nlay
        self.n1 = n1
        self.n2 = n2 if n2 is not None else [1.0, 1.0, 1.0]
        self.d = d if d is not None else [0.0, 0.0, 0.0]
        self.n3 = n3

class ControlPar:
    num_cams: int
    img_base_name: List[str]
    cal_img_base_name: List[str]
    hp_flag: int
    allCam_flag: int
    tiff_flag: int
    imx: int
    imy: int
    pix_x: float
    pix_y: float
    chfield: int
    mm: MMNP

    def __init__(self, num_cams: int = 1) -> None:
        self.num_cams = num_cams
        self.img_base_name = [""] * num_cams
        self.cal_img_base_name = [""] * num_cams
        self.hp_flag = 0
        self.allCam_flag = 0
        self.tiff_flag = 0
        self.imx = 256 # pix
        self.imy = 256 # pix
        self.pix_x = 0.01
        self.pix_y = 0.01
        self.chfield = 0
        self.mm = MMNP()

    def set_image_size(self, size):
        """Set image size (imx, imy) from a (width, height) tuple."""
        self.imx, self.imy = size

    def set_pixel_size(self, pix_size):
        """Set pixel size (pix_x, pix_y) from a (pix_x, pix_y) tuple."""
        self.pix_x, self.pix_y = pix_size

def read_control_par(filename: str) -> ControlPar:
    with open(filename, "r") as par_file:
        num_cams = int(par_file.readline().strip())
        ret = ControlPar(num_cams)
        for cam in range(num_cams):
            ret.img_base_name[cam] = par_file.readline().strip()
            ret.cal_img_base_name[cam] = par_file.readline().strip()
        ret.hp_flag = int(par_file.readline().strip())
        ret.allCam_flag = int(par_file.readline().strip())
        ret.tiff_flag = int(par_file.readline().strip())
        ret.imx = int(par_file.readline().strip())
        ret.imy = int(par_file.readline().strip())
        ret.pix_x = float(par_file.readline().strip())
        ret.pix_y = float(par_file.readline().strip())
        ret.chfield = int(par_file.readline().strip())
        ret.mm.n1 = float(par_file.readline().strip())
        ret.mm.n2[0] = float(par_file.readline().strip())
        ret.mm.n3 = float(par_file.readline().strip())
        ret.mm.d[0] = float(par_file.readline().strip())
    ret.mm.nlay = 1
    return ret

def free_control_par(cp: ControlPar) -> None:
    del cp

def compare_control_par(c1: ControlPar, c2: ControlPar) -> bool:
    if c1.num_cams != c2.num_cams:
        return False
    for cam in range(c1.num_cams):
        if c1.img_base_name[cam] != c2.img_base_name[cam]:
            return False
        if c1.cal_img_base_name[cam] != c2.cal_img_base_name[cam]:
            return False
    if (c1.hp_flag != c2.hp_flag or c1.allCam_flag != c2.allCam_flag or
            c1.tiff_flag != c2.tiff_flag or c1.imx != c2.imx or
            c1.imy != c2.imy or c1.pix_x != c2.pix_x or c1.pix_y != c2.pix_y or
            c1.chfield != c2.chfield):
        return False
    return compare_mm_np(c1.mm, c2.mm)

def compare_mm_np(mm_np1: MMNP, mm_np2: MMNP) -> bool:
    if mm_np1.n2[0] != mm_np2.n2[0] or mm_np1.d[0] != mm_np2.d[0]:
        return False
    if mm_np1.nlay != mm_np2.nlay or mm_np1.n1 != mm_np2.n1 or mm_np1.n3 != mm_np2.n3:
        return False
    return True

class TargetPar:
    discont: int
    gvthres: List[int]
    nnmin: int
    nnmax: int
    nxmin: int
    nxmax: int
    nymin: int
    nymax: int
    sumg_min: int
    cr_sz: int
    def __init__(self) -> None:
        self.discont = 0
        self.gvthres = [0, 0, 0, 0]
        self.nnmin = 0
        self.nnmax = 0
        self.nxmin = 0
        self.nxmax = 0
        self.nymin = 0
        self.nymax = 0
        self.sumg_min = 0
        self.cr_sz = 0

def read_target_par(filename: str = "target.par") -> TargetPar:
    ret = TargetPar()
    with open(filename, "r") as file:
        ret.gvthres[0] = int(file.readline().strip())
        ret.gvthres[1] = int(file.readline().strip())
        ret.gvthres[2] = int(file.readline().strip())
        ret.gvthres[3] = int(file.readline().strip())
        ret.discont = int(file.readline().strip())
        ret.nnmin, ret.nnmax = map(int, file.readline().strip().split())
        ret.nxmin, ret.nxmax = map(int, file.readline().strip().split())
        ret.nymin, ret.nymax = map(int, file.readline().strip().split())
        ret.sumg_min = int(file.readline().strip())
        ret.cr_sz = int(file.readline().strip())
    return ret

def compare_target_par(targ1: TargetPar, targ2: TargetPar) -> bool:
    return (targ1.discont == targ2.discont and
            targ1.gvthres[0] == targ2.gvthres[0] and
            targ1.gvthres[1] == targ2.gvthres[1] and
            targ1.gvthres[2] == targ2.gvthres[2] and
            targ1.gvthres[3] == targ2.gvthres[3] and
            targ1.nnmin == targ2.nnmin and targ1.nnmax == targ2.nnmax and
            targ1.nxmin == targ2.nxmin and targ1.nxmax == targ2.nxmax and
            targ1.nymin == targ2.nymin and targ1.nymax == targ2.nymax and
            targ1.sumg_min == targ2.sumg_min and targ1.cr_sz == targ2.cr_sz)

def write_target_par(targ: TargetPar, filename: str) -> None:
    with open(filename, "w") as file:
        file.write(f"{targ.gvthres[0]}\n{targ.gvthres[1]}\n{targ.gvthres[2]}\n{targ.gvthres[3]}\n")
        file.write(f"{targ.discont}\n{targ.nnmin} {targ.nnmax}\n{targ.nxmin} {targ.nxmax}\n")
        file.write(f"{targ.nymin} {targ.nymax}\n{targ.sumg_min}\n{targ.cr_sz}\n")
