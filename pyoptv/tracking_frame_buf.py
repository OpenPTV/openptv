import numpy as np
import numba
from scipy.optimize import minimize
import matplotlib.pyplot as plt

STR_MAX_LEN = 255
POSI = 4
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 4

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

class PathInfo:
    def __init__(self, prev, next, prio, finaldecis, inlist, x, decis, linkdecis):
        self.prev = prev
        self.next = next
        self.prio = prio
        self.finaldecis = finaldecis
        self.inlist = inlist
        self.x = x
        self.decis = decis
        self.linkdecis = linkdecis

class Frame:
    def __init__(self, num_cams, max_targets):
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.num_parts = 0
        self.path_info = [PathInfo(PREV_NONE, NEXT_NONE, PRIO_DEFAULT, 1000000.0, 0, np.zeros(3), np.zeros(POSI), np.zeros(POSI)) for _ in range(max_targets)]
        self.correspond = [Corres(-1, [-1, -1, -1, -1]) for _ in range(max_targets)]
        self.targets = [[Target(-1, 0, 0, 0, 0, 0, 0, -1) for _ in range(max_targets)] for _ in range(num_cams)]
        self.num_targets = [0] * num_cams

@numba.jit(nopython=True)
def compare_targets(t1, t2):
    return (t1.pnr == t2.pnr and t1.x == t2.x and t1.y == t2.y and
            t1.n == t2.n and t1.nx == t2.nx and t1.ny == t2.ny and
            t1.sumg == t2.sumg and t1.tnr == t2.tnr)

def read_targets(file_base, frame_num):
    filein = f"{file_base}{frame_num:04d}_targets" if frame_num > 0 else f"{file_base}_targets"
    try:
        with open(filein, "r") as f:
            num_targets = int(f.readline().strip())
            buffer = []
            for _ in range(num_targets):
                data = list(map(float, f.readline().strip().split()))
                buffer.append(Target(int(data[0]), data[1], data[2], int(data[3]), int(data[4]), int(data[5]), int(data[6]), int(data[7])))
            return buffer
    except Exception as e:
        print(f"Error reading file {filein}: {e}")
        return -1

def write_targets(buffer, num_targets, file_base, frame_num):
    fileout = f"{file_base}{frame_num:04d}_targets" if frame_num > 0 else f"{file_base}_targets"
    try:
        with open(fileout, "w") as f:
            f.write(f"{num_targets}\n")
            for t in buffer:
                f.write(f"{t.pnr} {t.x:.4f} {t.y:.4f} {t.n} {t.nx} {t.ny} {t.sumg} {t.tnr}\n")
        return 1
    except Exception as e:
        print(f"Error writing file {fileout}: {e}")
        return 0

@numba.jit(nopython=True)
def compare_corres(c1, c2):
    return (c1.nr == c2.nr and c1.p[0] == c2.p[0] and
            c1.p[1] == c2.p[1] and c1.p[2] == c2.p[2] and
            c1.p[3] == c2.p[3])

@numba.jit(nopython=True)
def compare_path_info(p1, p2):
    if not (p1.prev == p2.prev and p1.next == p2.next and
            p1.prio == p2.prio and p1.finaldecis == p2.finaldecis and
            p1.inlist == p2.inlist and np.array_equal(p1.x, p2.x)):
        return False
    for iter in range(POSI):
        if p1.decis[iter] != p2.decis[iter] or p1.linkdecis[iter] != p2.linkdecis[iter]:
            return False
    return True

@numba.jit(nopython=True)
def register_link_candidate(self, fitness, cand):
    self.decis[self.inlist] = fitness
    self.linkdecis[self.inlist] = cand
    self.inlist += 1

@numba.jit(nopython=True)
def reset_links(self):
    self.prev = PREV_NONE
    self.next = NEXT_NONE
    self.prio = PRIO_DEFAULT

def read_path_frame(cor_buf, path_buf, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    try:
        with open(f"{corres_file_base}.{frame_num}", "r") as filein:
            num_points = int(filein.readline().strip())
            if linkage_file_base:
                with open(f"{linkage_file_base}.{frame_num}", "r") as linkagein:
                    linkagein.readline()
            if prio_file_base:
                with open(f"{prio_file_base}.{frame_num}", "r") as prioin:
                    prioin.readline()
            for i in range(num_points):
                if linkage_file_base:
                    linkage_data = list(map(float, linkagein.readline().strip().split()))
                    path_buf[i].prev = int(linkage_data[0])
                    path_buf[i].next = int(linkage_data[1])
                else:
                    path_buf[i].prev = -1
                    path_buf[i].next = -2
                if prio_file_base:
                    prio_data = list(map(float, prioin.readline().strip().split()))
                    path_buf[i].prio = int(prio_data[5])
                else:
                    path_buf[i].prio = 4
                path_buf[i].inlist = 0
                path_buf[i].finaldecis = 1000000.0
                path_buf[i].decis = np.zeros(POSI)
                path_buf[i].linkdecis = np.full(POSI, -999)
                cor_data = list(map(float, filein.readline().strip().split()))
                path_buf[i].x = np.array(cor_data[1:4])
                cor_buf[i].p = list(map(int, cor_data[4:8]))
                cor_buf[i].nr = i
        return num_points
    except Exception as e:
        print(f"Error reading path frame: {e}")
        return -1

def write_path_frame(cor_buf, path_buf, num_parts, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    try:
        with open(f"{corres_file_base}.{frame_num}", "w") as corres_file:
            corres_file.write(f"{num_parts}\n")
            if linkage_file_base:
                with open(f"{linkage_file_base}.{frame_num}", "w") as linkage_file:
                    linkage_file.write(f"{num_parts}\n")
            if prio_file_base:
                with open(f"{prio_file_base}.{frame_num}", "w") as prio_file:
                    prio_file.write(f"{num_parts}\n")
            for i in range(num_parts):
                if linkage_file_base:
                    linkage_file.write(f"{path_buf[i].prev} {path_buf[i].next} {path_buf[i].x[0]:.3f} {path_buf[i].x[1]:.3f} {path_buf[i].x[2]:.3f}\n")
                corres_file.write(f"{i + 1} {path_buf[i].x[0]:.3f} {path_buf[i].x[1]:.3f} {path_buf[i].x[2]:.3f} {cor_buf[i].p[0]} {cor_buf[i].p[1]} {cor_buf[i].p[2]} {cor_buf[i].p[3]}\n")
                if prio_file_base:
                    prio_file.write(f"{path_buf[i].prev} {path_buf[i].next} {path_buf[i].x[0]:.3f} {path_buf[i].x[1]:.3f} {path_buf[i].x[2]:.3f} {path_buf[i].prio}\n")
        return 1
    except Exception as e:
        print(f"Error writing path frame: {e}")
        return 0

def frame_init(new_frame, num_cams, max_targets):
    new_frame.path_info = [PathInfo(PREV_NONE, NEXT_NONE, PRIO_DEFAULT, 1000000.0, 0, np.zeros(3), np.zeros(POSI), np.zeros(POSI)) for _ in range(max_targets)]
    new_frame.correspond = [Corres(-1, [-1, -1, -1, -1]) for _ in range(max_targets)]
    new_frame.targets = [[Target(-1, 0, 0, 0, 0, 0, 0, -1) for _ in range(max_targets)] for _ in range(num_cams)]
    new_frame.num_targets = [0] * num_cams
    new_frame.num_cams = num_cams
    new_frame.max_targets = max_targets
    new_frame.num_parts = 0

def free_frame(self):
    self.path_info = None
    self.correspond = None
    self.num_targets = None
    self.targets = None

def read_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
    self.num_parts = read_path_frame(self.correspond, self.path_info, corres_file_base, linkage_file_base, prio_file_base, frame_num)
    if self.num_parts == -1:
        return 0
    for cam in range(self.num_cams):
        self.num_targets[cam] = read_targets(target_file_base[cam], frame_num)
        if self.num_targets[cam] == -1:
            return 0
    return 1

def write_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
    status = write_path_frame(self.correspond, self.path_info, self.num_parts, corres_file_base, linkage_file_base, prio_file_base, frame_num)
    if status == 0:
        return 0
    for cam in range(self.num_cams):
        status = write_targets(self.targets[cam], self.num_targets[cam], target_file_base[cam], frame_num)
        if status == 0:
            return 0
    return 1

class FrameBufferBase:
    def __init__(self, buf_len, num_cams, max_targets):
        self.buf_len = buf_len
        self.num_cams = num_cams
        self._ring_vec = [Frame(num_cams, max_targets) for _ in range(buf_len * 2)]
        self.buf = self._ring_vec[:buf_len]
        self._vptr = None

    def free(self):
        self._vptr.free(self)

    def read_frame_at_end(self, frame_num, read_links):
        return self._vptr.read_frame_at_end(self, frame_num, read_links)

    def write_frame_from_start(self, frame_num):
        return self._vptr.write_frame_from_start(self, frame_num)

def fb_base_init(new_buf, buf_len, num_cams, max_targets):
    new_buf.buf_len = buf_len
    new_buf.num_cams = num_cams
    new_buf._ring_vec = [Frame(num_cams, max_targets) for _ in range(buf_len * 2)]
    new_buf.buf = new_buf._ring_vec[:buf_len]
    new_buf._vptr = None

def fb_base_free(self):
    self.buf = self._ring_vec[:self.buf_len]
    for frame in self.buf:
        free_frame(frame)
    self.buf = None
    self._ring_vec = None
    self._vptr = None

class FrameBuffer(FrameBufferBase):
    def __init__(self, buf_len, num_cams, max_targets, corres_file_base, linkage_file_base, prio_file_base, target_file_base):
        super().__init__(buf_len, num_cams, max_targets)
        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base
        self._vptr = self

    def free(self):
        fb_base_free(self)

    def read_frame_at_end(self, frame_num, read_links):
        if read_links:
            return read_frame(self.buf[-1], self.corres_file_base, self.linkage_file_base, self.prio_file_base, self.target_file_base, frame_num)
        else:
            return read_frame(self.buf[-1], self.corres_file_base, None, None, self.target_file_base, frame_num)

    def write_frame_from_start(self, frame_num):
        return write_frame(self.buf[0], self.corres_file_base, self.linkage_file_base, self.prio_file_base, self.target_file_base, frame_num)

def fb_next(self):
    self.buf = self.buf[1:] + self.buf[:1]

def fb_prev(self):
    self.buf = self.buf[-1:] + self.buf[:-1]
