import numpy as np

# For point positions
from .vec_utils import vec3d

POSI = 80
STR_MAX_LEN = 255
PT_UNUSED = -999
CORRES_NONE = -1
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 2

class Target:
    def __init__(self, pnr, x, y, n, nx, ny, sumg, tnr):
        self.pnr = pnr
        self.x = x
        self.y = y
        self.n = n
        self.nx = nx
        self.ny = ny
        self.sumg = sumg
        self.tnr = tnr

class Corres:
    def __init__(self, nr, p):
        self.nr = nr
        self.p = p

class P:
    def __init__(self, x, prev, next, prio):
        self.x = x
        self.prev = prev
        self.next = next
        self.prio = prio
        self.decis = np.zeros(POSI, dtype=np.float32)
        self.finaldecis = 0.0
        self.linkdecis = np.zeros(POSI, dtype=np.int32)
        self.inlist = 0

    def register_link_candidate(self, fitness, cand):
        self.decis[self.inlist] = fitness
        self.linkdecis[self.inlist] = cand
        self.inlist += 1

    def reset_links(self):
        self.prev = PREV_NONE
        self.next = NEXT_NONE
        self.prio = PRIO_DEFAULT
        self.inlist = 0
        self.finaldecis = 0.0

class Frame:
    def __init__(self, num_cams, max_targets):
        self.path_info = None
        self.correspond = None
        self.targets = None
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.num_parts = 0
        self.num_targets = np.zeros(num_cams, dtype=np.int32)

    def read_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
        self.correspond, self.path_info = read_path_frame(corres_file_base, linkage_file_base, prio_file_base, frame_num)
        self.targets, self.num_parts = read_targets(target_file_base, frame_num)
        return self

    def write_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
        write_path_frame(self.correspond, self.path_info, self.num_parts, corres_file_base, linkage_file_base, prio_file_base, frame_num)
        write_targets(self.targets, self.num_targets, target_file_base, frame_num)


class FrameBufBase:
    def __init__(self, buf_len, num_cams):
        self._vptr = None
        self.buf = [None] * buf_len
        self._ring_vec = [None] * (2 * buf_len)
        self.buf_len = buf_len
        self.num_cams = num_cams

    def fb_next(self):
        pass

    def fb_prev(self):
        pass


class FrameBuf(FrameBufBase):
    def __init__(self, buf_len, num_cams, corres_file_base, linkage_file_base, prio_file_base, target_file_base):
        super().__init__(buf_len, num_cams)
        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base

    def fb_read_frame_at_end(self, frame_num, read_links):
        pass

    def fb_write_frame_from_start(self, frame_num):
        pass

    def fb_free(self):
        pass


def fb_base_init(new_buf, buf_len, num_cams, max_targets):
    pass


def fb_init(new_buf, buf_len, num_cams, max_targets, corres_file_base, linkage_file_base, prio_file_base, target_file_base):
    pass


def fb_disk_free(self):
    pass


def fb_disk_read_frame_at_end(self, frame_num, read_links):
    pass


def fb_disk_write_frame_from_start(self, frame_num):
    pass
