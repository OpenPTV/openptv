
class tracking_run:
    def __init__(self):
        self.fb = None  # type: framebuf_base
        self.seq_par = None  # type: sequence_par
        self.tpar = None  # type: track_par
        self.vpar = None  # type: volume_par
        self.cpar = None  # type: control_par
        self.cal = None  # type: List[Calibration]
        self.flatten_tol = 0.0  # type: float
        self.ymin = 0.0  # type: float
        self.ymax = 0.0  # type: float
        self.lmax = 0.0  # type: float
        self.npart = 0  # type: int
        self.nlinks = 0  # type: int


def tr_new_legacy(seq_par_fname, tpar_fname, vpar_fname, cpar_fname, cal):
    cpar = read_control_par(cpar_fname)
    seq_par = read_sequence_par(seq_par_fname, cpar.num_cams)
    return tr_new(seq_par, read_track_par(tpar_fname), 
        read_volume_par(vpar_fname), cpar, 4, 20000,
        "res/rt_is", "res/ptv_is", "res/added", cal, 10000)

def tr_new(seq_par, tpar, vpar, cpar, buf_len, max_targets,
           corres_file_base, linkage_file_base, prio_file_base, cal, flatten_tol):
    tr = tracking_run()
    tr.tpar = tpar
    tr.vpar = vpar
    tr.cpar = cpar
    tr.seq_par = seq_par
    tr.cal = cal
    tr.flatten_tol = flatten_tol
    
    tr.fb = framebuf_base()
    fb_init(tr.fb, buf_len, cpar.num_cams, max_targets,
        corres_file_base, linkage_file_base, prio_file_base, seq_par.img_base_name)
    
    tr.lmax = norm(tpar.dvxmin - tpar.dvxmax, tpar.dvymin - tpar.dvymax, tpar.dvzmin - tpar.dvzmax)
    volumedimension(vpar.X_lay[1], vpar.X_lay[0], tr.ymax, tr.ymin, vpar.Zmax_lay[1], vpar.Zmin_lay[0],
                    vpar, cpar, cal)
    
    tr.npart = 0
    tr.nlinks = 0
    
    return tr

def tr_free(tr):
    free(tr.fb)
    free(tr.seq_par.img_base_name)
    free(tr.seq_par)
    free(tr.tpar)
    free(tr.vpar)
    free_control_par(tr.cpar)
