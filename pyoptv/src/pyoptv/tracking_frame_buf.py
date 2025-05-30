# type: ignore
import numpy as np
from typing import Any, List
from .constants import MAX_TARGETS
from .constants import POSI, PREV_NONE, NEXT_NONE, PRIO_DEFAULT, MAX_TARGETS

class Target:
    """
    Represents a detected target (particle) in an image.
    """
    pnr: int
    x: float
    y: float
    n: int
    nx: int
    ny: int
    sumg: float
    tnr: int

    def __init__(
        self,
        pnr: int = -1,
        x: float = 0.0,
        y: float = 0.0,
        n: int = 0,
        nx: int = 0,
        ny: int = 0,
        sumg: float = 0.0,
        tnr: int = -1
    ) -> None:
        self.pnr = pnr
        self.x = x
        self.y = y
        self.n = n
        self.nx = nx
        self.ny = ny
        self.sumg = sumg
        self.tnr = tnr

class Corres:
    """
    Correspondence information for a 3D particle across cameras.
    """
    nr: int
    p: List[int]

    def __init__(self, nr: int, p: List[int]) -> None:
        self.nr = nr
        self.p = p

class PathInfo:
    """
    Path information for a tracked particle.
    """
    prev: int
    next: int
    prio: int
    finaldecis: float
    inlist: int
    x: np.ndarray
    decis: np.ndarray
    linkdecis: np.ndarray

    def __init__(self, prev: int, next: int, prio: int, finaldecis: float, inlist: int, x: np.ndarray, decis: np.ndarray, linkdecis: np.ndarray) -> None:
        self.prev = prev
        self.next = next
        self.prio = prio
        self.finaldecis = finaldecis
        self.inlist = inlist
        self.x = x
        self.decis = decis
        self.linkdecis = linkdecis

    def register_link_candidate(self, fitness: float, cand: int) -> None:
        """
        Register a candidate link in the path info structure.
        """
        self.decis[self.inlist] = fitness
        self.linkdecis[self.inlist] = cand
        self.inlist += 1

    def reset_links(self) -> None:
        """
        Reset the link information for this path.
        """
        self.prev = PREV_NONE
        self.next = NEXT_NONE
        self.prio = PRIO_DEFAULT

class Frame:
    """
    Represents a frame in the tracking buffer, holding all targets, correspondences, and path info.
    """
    num_cams: int
    max_targets: int
    num_parts: int
    path_info: List[PathInfo]
    correspond: List[Corres]
    targets: List[List[Target]]
    num_targets: List[int]

    def __init__(self, num_cams: int, max_targets: int = None) -> None:
        if max_targets is None:
            max_targets = MAX_TARGETS
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.num_parts = 0
        self.path_info = [PathInfo(PREV_NONE, NEXT_NONE, PRIO_DEFAULT, 1000000.0, 0, np.zeros(3), np.zeros(POSI), np.zeros(POSI)) for _ in range(max_targets)]
        self.correspond = [Corres(-1, [-1, -1, -1, -1]) for _ in range(max_targets)]
        self.targets = [[Target(-1, 0, 0, 0, 0, 0, 0, -1) for _ in range(max_targets)] for _ in range(num_cams)]
        self.num_targets = [0] * num_cams


def compare_targets(t1: Target, t2: Target) -> bool:
    return (
        t1.pnr == t2.pnr and t1.x == t2.x and t1.y == t2.y and
        t1.n == t2.n and t1.nx == t2.nx and t1.ny == t2.ny and
        t1.sumg == t2.sumg and t1.tnr == t2.tnr
    )

def read_targets(file_base: str, frame_num: int) -> List[Target]:
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

def write_targets(buffer: List[Target], num_targets: int, file_base: str, frame_num: int) -> int:
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
def compare_corres(c1: Corres, c2: Corres) -> bool:
    return (c1.nr == c2.nr and c1.p[0] == c2.p[0] and
            c1.p[1] == c2.p[1] and c1.p[2] == c2.p[2] and
            c1.p[3] == c2.p[3])
def compare_path_info(p1: PathInfo, p2: PathInfo) -> bool:
    if not (p1.prev == p2.prev and p1.next == p2.next and
            p1.prio == p2.prio and p1.finaldecis == p2.finaldecis and
            p1.inlist == p2.inlist and np.array_equal(p1.x, p2.x)):
        return False
    for iter in range(POSI):
        if p1.decis[iter] != p2.decis[iter] or p1.linkdecis[iter] != p2.linkdecis[iter]:
            return False
    return True
def read_path_frame(
    cor_buf: List[Corres],
    path_buf: List[PathInfo],
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int
) -> int:
    """
    Read a path frame from disk into correspondence and path buffers.
    """
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

def write_path_frame(
    cor_buf: List[Corres],
    path_buf: List[PathInfo],
    num_parts: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    frame_num: int
) -> int:
    """
    Write a path frame to disk from correspondence and path buffers.
    """
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

def frame_init(new_frame: Frame, num_cams: int, max_targets: int) -> None:
    """
    Initialize a Frame object with the given number of cameras and targets.
    """
    new_frame.path_info = [PathInfo(PREV_NONE, NEXT_NONE, PRIO_DEFAULT, 1000000.0, 0, np.zeros(3), np.zeros(POSI), np.zeros(POSI)) for _ in range(max_targets)]
    new_frame.correspond = [Corres(-1, [-1, -1, -1, -1]) for _ in range(max_targets)]
    new_frame.targets = [[Target(-1, 0, 0, 0, 0, 0, 0, -1) for _ in range(max_targets)] for _ in range(num_cams)]
    new_frame.num_targets = [0] * num_cams
    new_frame.num_cams = num_cams
    new_frame.max_targets = max_targets
    new_frame.num_parts = 0

def free_frame(frame: Frame) -> None:
    """
    Free the memory associated with a Frame object.
    """
    frame.path_info = None
    frame.correspond = None
    frame.num_targets = None
    frame.targets = None

def read_frame(
    frame: Frame,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    target_file_base: List[str],
    frame_num: int
) -> int:
    """
    Read a frame from disk into a Frame object.
    """
    frame.num_parts = read_path_frame(frame.correspond, frame.path_info, corres_file_base, linkage_file_base, prio_file_base, frame_num)
    if frame.num_parts == -1:
        return 0
    for cam in range(frame.num_cams):
        frame.num_targets[cam] = read_targets(target_file_base[cam], frame_num)
        if frame.num_targets[cam] == -1:
            return 0
    return 1

def write_frame(
    self: Frame,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    target_file_base: List[str],
    frame_num: int
) -> int:
    """
    Write a Frame object to disk.
    """
    status = write_path_frame(self.correspond, self.path_info, self.num_parts, corres_file_base, linkage_file_base, prio_file_base, frame_num)
    if status == 0:
        return 0
    for cam in range(self.num_cams):
        status = write_targets(self.targets[cam], self.num_targets[cam], target_file_base[cam], frame_num)
        if status == 0:
            return 0
    return 1

class FrameBufferBase:
    """
    Base class for frame buffer objects, implements ring buffer logic.
    """
    def __init__(self, buf_len: int, num_cams: int, max_targets: int) -> None:
        self.buf_len = buf_len
        self.num_cams = num_cams
        self._ring_vec = [Frame(num_cams, max_targets) for _ in range(buf_len * 2)]
        self.buf = self._ring_vec[:buf_len]
        self._vptr = None

    def free(self) -> None:
        """
        Free the frame buffer.
        """
        self._vptr.free(self)

    def read_frame_at_end(self, frame_num: int, read_links: bool) -> int:
        """
        Read a frame at the end of the buffer.
        """
        return self._vptr.read_frame_at_end(self, frame_num, read_links)

    def write_frame_from_start(self, frame_num: int) -> int:
        """
        Write a frame from the start of the buffer.
        """
        return self._vptr.write_frame_from_start(self, frame_num)

def fb_base_init(new_buf: FrameBufferBase, buf_len: int, num_cams: int, max_targets: int) -> None:
    """
    Initialize a FrameBufferBase object.
    """
    new_buf.buf_len = buf_len
    new_buf.num_cams = num_cams
    new_buf._ring_vec = [Frame(num_cams, max_targets) for _ in range(buf_len * 2)]
    new_buf.buf = new_buf._ring_vec[:buf_len]
    new_buf._vptr = None

def fb_base_free(fb: FrameBufferBase) -> None:
    """
    Free the memory associated with a FrameBufferBase object.
    """
    fb.buf = fb._ring_vec[:fb.buf_len]
    for frame in fb.buf:
        free_frame(frame)
    fb.buf = None
    fb._ring_vec = None
    fb._vptr = None

class FrameBuffer(FrameBufferBase):
    """
    Frame buffer for tracking, with file I/O support.
    """
    def __init__(
        self,
        buf_len: int,
        num_cams: int,
        max_targets: int,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        target_file_base: List[str]
    ) -> None:
        super().__init__(buf_len, num_cams, max_targets)
        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base
        self._vptr = self

    def free(self) -> None:
        """
        Free the frame buffer.
        """
        fb_base_free(self)

    def read_frame_at_end(self, frame_num: int, read_links: bool) -> int:
        """
        Read a frame at the end of the buffer.
        """
        if read_links:
            return read_frame(self.buf[-1], self.corres_file_base, self.linkage_file_base, self.prio_file_base, self.target_file_base, frame_num)
        else:
            return read_frame(self.buf[-1], self.corres_file_base, None, None, self.target_file_base, frame_num)

    def write_frame_from_start(self, frame_num: int) -> int:
        """
        Write a frame from the start of the buffer.
        """
        return write_frame(self.buf[0], self.corres_file_base, self.linkage_file_base, self.prio_file_base, self.target_file_base, frame_num)

    def fb_next(self) -> None:
        """
        Advance the buffer to the next frame.
        """
        self.buf = self.buf[1:] + self.buf[:1]

    def fb_prev(self) -> None:
        """
        Move the buffer to the previous frame.
        """
        self.buf = self.buf[-1:] + self.buf[:-1]

def fb_write_frame_from_start(fb: FrameBuffer, frame_num: int) -> int:
    """
    Write a frame from the start of the buffer using the FrameBuffer interface.
    """
    return fb.write_frame_from_start(frame_num)

def fb_read_frame_at_end(fb: FrameBuffer, frame_num: int, read_links: bool) -> int:
    """
    Read a frame at the end of the buffer using the FrameBuffer interface.
    """
    return fb.read_frame_at_end(frame_num, read_links)

def fb_next(fb: FrameBuffer) -> None:
    """
    Advance the buffer to the next frame (C API compatibility).
    """
    fb.fb_next()

def fb_prev(fb: FrameBuffer) -> None:
    """
    Move the buffer to the previous frame (C API compatibility).
    """
    fb.fb_prev()
