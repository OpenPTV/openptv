import numpy as np
from typing import List
from scipy.ndimage import label
import matplotlib.pyplot as plt
from pyoptv.tracking_frame_buf import Target
from pyoptv.parameters import ControlPar, TargetPar


class Peak:
    def __init__(
        self,
        pos: int = 0,
        status: int = 0,
        xmin: int = 0,
        xmax: int = 0,
        ymin: int = 0,
        ymax: int = 0,
        n: int = 0,
        sumg: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        unr: int = 0,
        touch: list = None,
        n_touch: int = 0,
    ):
        self.pos = pos
        self.status = status
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.n = n
        self.sumg = sumg
        self.x = x
        self.y = y
        self.unr = unr
        self.touch = touch if touch is not None else [0, 0, 0, 0]
        self.n_touch = n_touch


def targ_rec(
    img: np.ndarray,
    targ_par: TargetPar,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    cpar: ControlPar,
    num_cam: int,
) -> List[Target]:
    imx, imy = cpar.imx, cpar.imy
    thres = targ_par.gvthres[num_cam]
    disco = targ_par.discont
    img0 = np.copy(img)
    img0 = np.pad(img0, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    xmin, xmax = max(1, xmin), min(imx - 1, xmax)
    ymin, ymax = max(1, ymin), min(imy - 1, ymax)
    targets: List[Target] = []

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            gv = img0[i, j]
            if gv > thres and all(
                gv >= img0[i + di, j + dj]
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                if di != 0 or dj != 0
            ):
                yn, xn = i, j
                sumg = gv
                img0[i, j] = 0
                xa, xb, ya, yb = xn, xn, yn, yn
                gv -= thres
                x, y = xn * gv, yn * gv
                numpix = 1
                waitlist = [(j, i)]

                while waitlist:
                    gvref = img[waitlist[0][1], waitlist[0][0]]
                    x4 = [
                        waitlist[0][0] - 1,
                        waitlist[0][0] + 1,
                        waitlist[0][0],
                        waitlist[0][0],
                    ]
                    y4 = [
                        waitlist[0][1],
                        waitlist[0][1],
                        waitlist[0][1] - 1,
                        waitlist[0][1] + 1,
                    ]

                    for xn, yn in zip(x4, y4):
                        if not (
                            xmin - 1 < xn < xmax + 1 and ymin - 1 < yn < ymax + 1
                        ):
                            continue
                        gv = img0[yn, xn]
                        if (
                            gv > thres
                            and gv <= gvref + disco
                            and all(
                                gvref + disco
                                >= img[yn + di, xn + dj]
                                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            )
                        ):
                            sumg += gv
                            img0[yn, xn] = 0
                            xa, xb, ya, yb = (
                                min(xa, xn),
                                max(xb, xn),
                                min(ya, yn),
                                max(yb, yn),
                            )
                            waitlist.append((xn, yn))
                            x += xn * (gv - thres)
                            y += yn * (gv - thres)
                            numpix += 1

                    waitlist.pop(0)

                if (
                    xa == xmin - 1
                    or ya == ymin - 1
                    or xb == xmax + 1
                    or yb == ymax + 1
                ):
                    continue

                nx, ny = xb - xa + 1, yb - ya + 1
                if (
                    targ_par.nnmin <= numpix <= targ_par.nnmax
                    and targ_par.nxmin <= nx <= targ_par.nxmax
                    and targ_par.nymin <= ny <= targ_par.nymax
                    and sumg > targ_par.sumg_min
                ):
                    sumg -= numpix * thres
                    x_c = x / sumg + 0.5
                    y_c = y / sumg + 0.5
                    targets.append(
                        Target(
                            pnr=len(targets),
                            x=x_c,
                            y=y_c,
                            n=numpix,
                            nx=nx,
                            ny=ny,
                            sumg=sumg,
                            tnr=-1,  # Use -1 as default tnr, matching C struct default
                        )
                    )

    if not targets:
        targets.append(
            Target(
                pnr=1,
                x=1,
                y=1,
                n=1,
                nx=1,
                ny=1,
                sumg=1,
                tnr=-1,  # Use -1 as default tnr
            )
        )

    return targets


def peak_fit(
    img: np.ndarray,
    targ_par: TargetPar,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    cpar: ControlPar,
    num_cam: int,
) -> List[Target]:
    imx, imy = cpar.imx, cpar.imy
    thres = targ_par.gvthres[num_cam]
    disco = targ_par.discont
    label_img = np.zeros((imx, imy), dtype=np.int16)
    peaks: List[Peak] = []
    n_peaks = 0

    for i in range(ymin, ymax - 1):
        for j in range(xmin, xmax):
            n = i * imx + j
            gv = img[i, j]
            if gv <= thres or label_img[i, j] != 0:
                continue
            if all(
                gv >= img[i + di, j + dj]
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                if di != 0 or dj != 0
            ):
                n_peaks += 1
                label_img[i, j] = n_peaks
                peaks.append(
                    Peak(
                        pos=n,
                        status=1,
                        xmin=j,
                        xmax=j,
                        ymin=i,
                        ymax=i,
                        unr=0,
                        n=0,
                        sumg=0,
                        x=0.0,
                        y=0.0,
                        n_touch=0,
                        touch=[0, 0, 0, 0],
                    )
                )
                waitlist = [(j, i)]

                while waitlist:
                    gvref = img[waitlist[0][1], waitlist[0][0]]
                    x8 = [
                        waitlist[0][0] - 1,
                        waitlist[0][0] + 1,
                        waitlist[0][0],
                        waitlist[0][0],
                    ]
                    y8 = [
                        waitlist[0][1],
                        waitlist[0][1],
                        waitlist[0][1] - 1,
                        waitlist[0][1] + 1,
                    ]

                    for xn, yn in zip(x8, y8):
                        if (
                            not (0 <= xn < imx and 0 <= yn < imy)
                            or label_img[yn, xn] != 0
                        ):
                            continue
                        gv = img[yn, xn]
                        if (
                            gv > thres
                            and gv <= gvref + disco
                            and all(
                                gvref + disco
                                >= img[yn + di, xn + dj]
                                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            )
                        ):
                            label_img[yn, xn] = n_peaks
                            waitlist.append((xn, yn))

                    waitlist.pop(0)

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            n = i * imx + j
            if label_img[i, j] > 0:
                pnr = label_img[i, j]
                gv = img[i, j]
                peak = peaks[pnr - 1]
                peak.n += 1
                peak.sumg += gv
                peak.x += j * gv
                peak.y += i * gv
                peak.xmin, peak.xmax = min(peak.xmin, j), max(peak.xmax, j)
                peak.ymin, peak.ymax = min(peak.ymin, i), max(peak.ymax, i)

                for di, dj in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]:
                    if 0 <= i + di < imy and 0 <= j + dj < imx:
                        check_touch(peak, pnr, label_img[i + di, j + dj])

    for i in range(n_peaks):
        if peaks[i].n_touch == 0 or peaks[i].unr != 0:
            continue
        x1, y1 = peaks[i].x / peaks[i].sumg, peaks[i].y / peaks[i].sumg
        gv1 = img[peaks[i].pos // imx, peaks[i].pos % imx]

        for j in range(peaks[i].n_touch):
            p2 = peaks[i].touch[j] - 1
            if p2 >= n_peaks or p2 < 0 or peaks[p2].unr != 0:
                continue
            x2, y2 = peaks[p2].x / peaks[p2].sumg, peaks[p2].y / peaks[p2].sumg
            gv2 = img[peaks[p2].pos // imx, peaks[p2].pos % imx]
            s12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            unify = (
                1
                if s12 < 2.0
                else all(
                    img[
                        int(y1 + l * (y2 - y1) / s12),
                        int(x1 + l * (x2 - x1) / s12),
                    ]
                    + disco
                    >= gv1 + l * (gv2 - gv1) / s12
                    for l in range(1, int(s12))
                )
            )

            if unify:
                peaks[i].unr = p2
                peaks[p2].x += peaks[i].x
                peaks[p2].y += peaks[i].y
                peaks[p2].sumg += peaks[i].sumg
                peaks[p2].n += peaks[i].n
                peaks[p2].xmin, peaks[p2].xmax = min(
                    peaks[p2].xmin, peaks[i].xmin
                ), max(peaks[p2].xmax, peaks[i].xmax)
                peaks[p2].ymin, peaks[p2].ymax = min(
                    peaks[p2].ymin, peaks[i].ymin
                ), max(peaks[p2].ymax, peaks[i].ymax)

    targets: List[Target] = []
    for i in range(n_peaks):
        if (
            peaks[i].xmin == xmin
            and (xmax - xmin) > 32
            or peaks[i].ymin == ymin
            and (xmax - xmin) > 32
            or peaks[i].xmax == xmax - 1
            and (xmax - xmin) > 32
            or peaks[i].ymax == ymax - 1
            and (xmax - xmin) > 32
        ):
            continue
        if (
            peaks[i].unr == 0
            and peaks[i].sumg > targ_par.sumg_min
            and targ_par.nxmin
            <= peaks[i].xmax - peaks[i].xmin + 1
            <= targ_par.nxmax
            and targ_par.nymin
            <= peaks[i].ymax - peaks[i].ymin + 1
            <= targ_par.nymax
            and targ_par.nnmin <= peaks[i].n <= targ_par.nnmax
        ):
            sumg = peaks[i].sumg
            x, y = 0.5 + peaks[i].x / sumg, 0.5 + peaks[i].y / sumg
            targets.append(
                Target(
                    x=x,
                    y=y,
                    sumg=sumg,
                    n=peaks[i].n,
                    nx=peaks[i].xmax - peaks[i].xmin + 1,
                    ny=peaks[i].ymax - peaks[i].ymin + 1,
                    tnr=-1,
                    pnr=len(targets),
                )
            )

    return targets


def check_touch(peak: Peak, p1: int, p2: int) -> None:
    if p2 == 0 or p2 == p1:
        return
    if p2 not in peak.touch[: peak.n_touch]:
        peak.touch[peak.n_touch] = p2
        peak.n_touch = min(peak.n_touch + 1, 3)

