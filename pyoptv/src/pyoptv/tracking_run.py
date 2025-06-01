# type: ignore
import numpy as np
from typing import List, Optional, Tuple
from .tracking_frame_buf import FrameBuffer
from .parameters import (
    SequencePar, TrackPar, VolumePar, ControlPar,
    read_control_par, read_sequence_par, read_track_par, read_volume_par
)
from .calibration import Calibration

class TrackingRun:
    seq_par: SequencePar
    tpar: TrackPar
    vpar: VolumePar
    cpar: ControlPar
    cal: List[object]
    flatten_tol: float
    fb: FrameBuffer
    lmax: float
    ymax: float
    ymin: float
    npart: int
    nlinks: int

    def __init__(
        self,
        seq_par: SequencePar,
        tpar: TrackPar,
        vpar: VolumePar,
        cpar: ControlPar,
        cal: List[Calibration],
        buf_len: int,
        max_targets: int,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        flatten_tol: float,
    ) -> None:
        self.seq_par = seq_par
        self.tpar = tpar
        self.vpar = vpar
        self.cpar = cpar
        self.cal = cal
        self.flatten_tol = flatten_tol
        self.fb = FrameBuffer(
            buf_len,
            cpar.num_cams,
            max_targets,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            seq_par.img_base_name,
        )
        self.lmax = np.linalg.norm([
            tpar.dvxmin - tpar.dvxmax,
            tpar.dvymin - tpar.dvymax,
            tpar.dvzmin - tpar.dvzmax,
        ])
        self.ymax, self.ymin = self.volumedimension(
            vpar.X_lay[1], vpar.X_lay[0], vpar.Zmax_lay[1], vpar.Zmin_lay[0], vpar, cpar, cal
        )
        self.npart = 0
        self.nlinks = 0

    def volumedimension(
        self,
        X_lay1: float,
        X_lay0: float,
        Zmax_lay1: float,
        Zmin_lay0: float,
        vpar: VolumePar,
        cpar: ControlPar,
        cal: List[object],
    ) -> Tuple[float, float]:
        # Placeholder for the actual implementation
        return 0, 0

    @staticmethod
    def tr_new_legacy(
        seq_par_fname: str,
        tpar_fname: str,
        vpar_fname: str,
        cpar_fname: str,
        cal: List[object],
    ) -> 'TrackingRun':
        cpar = read_control_par(cpar_fname)
        seq_par = read_sequence_par(seq_par_fname, cpar.num_cams)
        return TrackingRun(
            seq_par,
            read_track_par(tpar_fname),
            read_volume_par(vpar_fname),
            cpar,
            cal,
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            10000,
        )

    def tr_free(self) -> None:
        del self.fb
        del self.seq_par.img_base_name
        del self.seq_par
        del self.tpar
        del self.vpar
        del self.cpar

