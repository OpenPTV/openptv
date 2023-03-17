import numpy as np

# For point positions
from .vec_utils import vec3d

# Define constants
POSI = 80
STR_MAX_LEN = 255
PT_UNUSED = -999
CORRES_NONE = -1
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 2

# Define struct target
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

# Define function compare_targets
def compare_targets(t1, t2):
    if t1.x < t2.x:
        return -1
    elif t1.x > t2.x:
        return 1
    else:
        return 0

# Define function read_targets
def read_targets(buffer, file_base, frame_num):
    # implementation not provided
    pass

# Define function write_targets
def write_targets(buffer, num_targets, file_base, frame_num):
    # implementation not provided
    pass

# Define struct corres
class Corres:
    def __init__(self, nr, p1, p2, p3, p4):
        self.nr = nr
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

# Define function compare_corres
def compare_corres(c1, c2):
    if c1.nr < c2.nr:
        return -1
    elif c1.nr > c2.nr:
        return 1
    else:
        return 0

# Define struct P
class P:
    def __init__(self, x, prev, next, prio):
        self.x = x
        self.prev = prev
        self.next = next
        self.prio = prio
        self.decis = [0.0]*POSI
        self.finaldecis = 0.0
        self.linkdecis = [-1]*POSI
        self.inlist = 0

# Define function compare_path_info
def compare_path_info(p1, p2):
    if p1.x.x < p2.x.x:
        return -1
    elif p1.x.x > p2.x.x:
        return 1
    else:
        return 0

# Define function register_link_candidate
def register_link_candidate(self, fitness, cand):
    self.decis = self.decis[:self.inlist] + [fitness] + self.decis[self.inlist:-1]
    self.linkdecis = self.linkdecis[:self.inlist] + [cand] + self.linkdecis[self.inlist:-1]
    self.inlist += 1

# Define function reset_links
def reset_links(self):
    self.inlist = 0
    self.decis = [0.0]*POSI
    self.finaldecis = 0.0
    self.linkdecis = [-1]*POSI

# Define function read_path_frame
def read_path_frame(cor_buf, path_buf, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    # implementation not provided
    pass

# Define function write_path_frame
def write_path_frame(cor_buf, path_buf, num_parts, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    # implementation not provided
    pass

# Define struct frame
class Frame:
    def __init__(self, targets, num_cams, max_targets, num_parts, num_targets):
        self.path_info = []
        self.correspond = []
        self.targets = targets
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.num_parts = num_parts
        self.num_targets = num_targets

# Define function frame_init
def frame_init(new_frame, num_cams, max_targets):
    new_frame.targets = [None] * num_cams
    new_frame.num_cams = num_cams
    new_frame.max_targets = max_targets
    new_frame.num_targets = [0] * num_cams

# Define function free_frame
def free_frame(self):
    self.path_info = []
    self.correspond = []
    self.targets = []
    self.num_cams = 0
    self.max_targets = 0
    self.num_parts = 0
    self.num_targets = []

# Define function read_frame
def read_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
    # implementation not provided
    pass

# Define function write_frame
def write_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
    # implementation not provided
    pass

# Define struct fb_vtable
class FB_VTABLE:
    def __init__(self, free, read_frame_at_end, write_frame_from_start):
        self.free = free
        self.read_frame_at_end = read_frame_at_end
        self.write_frame_from_start = write_frame_from_start

# Define struct framebuf_base
class FramebufBase:
    def __init__(self, buf_len, num_cams, max_targets):
        self._vptr = None
        self.buf = [None] * buf_len
        self._ring_vec = self.buf + buf_len
        self.buf_len = buf_len
        self.num_cams = num_cams

# Define function fb_free
def fb_free(self):
    self._vptr['free'](self)

# Define function fb_read_frame_at_end
def fb_read_frame_at_end(self, frame_num, read_links):
    return self._vptr['read_frame_at_end'](self, frame_num, read_links)

# Define function fb_write_frame_from_start
def fb_write_frame_from_start(self, frame_num):
    return self._vptr['write_frame_from_start'](self, frame_num)

# Define function fb_base_init
def fb_base_init(new_buf, buf_len, num_cams, max_targets):
    new_buf.buf = [None] * buf_len
    new_buf._ring_vec = new_buf.buf + buf_len
    new_buf.buf_len = buf_len
    new_buf.num_cams = num_cams

# Define function fb_next
def fb_next(self):
    self.buf = self.buf[1:] + self.buf[:1]

# Define function fb_prev
def fb_prev(self):
    self.buf = self.buf[-1:] + self.buf[:-1]

# Define struct framebuf
class Framebuf:
    def __init__(self, buf_len, num_cams, max_targets, corres_file_base, linkage_file_base, prio_file_base, target_file_base):
        self._vptr = FB_VTABLE(None, None, None)
        self.base = FramebufBase(buf_len, num_cams, max_targets)
        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base

# Define function fb_init
def fb_init(new_buf, buf_len, num_cams, max_targets, corres_file_base, linkage_file_base, prio_file_base, target_file_base):
    new_buf._vptr = FB_VTABLE(fb_disk_free, fb_disk_read_frame_at_end, fb_disk_write_frame_from_start)
    fb_base_init(new_buf.base, buf_len, num_cams, max_targets)
    new_buf.corres_file_base = corres_file_base
    new_buf.linkage_file_base = linkage_file_base
    new_buf.prio_file_base = prio_file_base
    new_buf.target_file_base = target_file_base

# Define function fb_disk_free
def fb_disk_free(self):
    # implementation not provided
    pass

# Define function fb_disk_read_frame_at_end
def fb_disk_read_frame_at_end(self, frame_num, read_links):
    # implementation not provided
    pass

# Define function fb_disk_write_frame_from_start
def fb_disk_write_frame_from_start(self, frame_num):
    # implementation not provided
    pass

def compare_targets(t1, t2):
    return (t1.pnr == t2.pnr) and (t1.x == t2.x) and (t1.y == t2.y) and \
           (t1.n == t2.n) and (t1.nx == t2.nx) and (t1.ny == t2.ny) and \
           (t1.sumg == t2.sumg) and (t1.tnr == t2.tnr)

import os

# Step 2: Define the function read_targets()
def read_targets(buffer, file_base, frame_num):
    STR_MAX_LEN = 1000
    filein = ""
    tix = 0
    num_targets = 0
    scanf_ok = 0

    # Step 3: Check if frame_num > 0, construct filein string accordingly
    if frame_num > 0:
        filein = "%s%04d%s"%(file_base, frame_num, "_targets")
    else:
        filein = file_base + "_targets"

    # Step 4: Open the filein for reading
    try:
        FILEIN = open(filein, "r")

    # Step 5: Handle the exceptions
    except:
        print("Can't open ascii file: ", filein)
        return -1

    # Step 6: Read the first line and obtain the number of targets
    num_targets = int(FILEIN.readline())

    # Step 7: Loop through the remaining lines of the file to read the target data
    for tix in range(num_targets):
        data = FILEIN.readline().split()
        
        # Step 8: Check if the file is correctly formatted, i.e., eight fields of data
        if len(data) != 8:
            print("Bad format for file: ", filein)
            FILEIN.close()
            return -1

        # Step 9: Assign the values to the target buffer
        buffer[tix].pnr = int(data[0])
        buffer[tix].x = float(data[1])
        buffer[tix].y = float(data[2])
        buffer[tix].n = int(data[3])
        buffer[tix].nx = int(data[4])
        buffer[tix].ny = int(data[5])
        buffer[tix].sumg = int(data[6])
        buffer[tix].tnr = int(data[7])

    # Step 10: Close the file and return the number of targets read
    FILEIN.close()
    return num_targets

def write_targets(target_buffer, num_targets, file_base, frame_num):
    
    # Step 3: Define Required Variables
    tix, print_ok, success = 0, 0, 0
    fileout = ""
    STR_MAX_LEN = 200
    
    # Step 4: Check Frame Number and Define File Output
    if frame_num == 0:
        fileout = file_base + "_targets"
    else:
        fileout = "{}{:04d}_targets".format(file_base, frame_num)
    
    # Step 5: Open File in Write Mode
    with open(fileout, 'w') as FILEOUT:
        
        # Step 6: Write Number of Targets to File
        if FILEOUT.write("{}\n".format(num_targets)) <= 0:
            print("Write error in file {}".format(fileout))
                
        # Step 7: Write Targets to File
        for tix in range(num_targets):
            print_ok = FILEOUT.write("{:4d} {:9.4f} {:9.4f} {:5d} {:5d} {:5d} {:5d} {:5d}\n".format(
                    target_buffer[tix].pnr, target_buffer[tix].x, target_buffer[tix].y, target_buffer[tix].n, 
                    target_buffer[tix].nx, target_buffer[tix].ny, target_buffer[tix].sumg, target_buffer[tix].tnr))
            
            if print_ok <= 0:
                print("Write error in file {}".format(fileout))
                break
            
        # Step 8: Set the Function Output for Success or Failure            
        if print_ok > 0:
            success = 1
    
    return success

def compare_corres(c1, c2):
    return ((c1.nr == c2.nr) and (c1.p[0] == c2.p[0]) and \
        (c1.p[1] == c2.p[1]) and (c1.p[2] == c2.p[2]) and \
        (c1.p[3] == c2.p[3]))

def compare_path_info(p1, p2):
    iter = 0
    if not ((p1.prev == p2.prev) and (p1.next == p2.next) and \
        (p1.prio == p2.prio) and (p1.finaldecis == p2.finaldecis) and \
        (p1.inlist == p2.inlist) and vec_cmp(p1.x, p2.x)):
        return 0

    while iter < POSI:
        if p1.decis[iter] != p2.decis[iter]: 
            return 0
        if p1.linkdecis[iter] != p2.linkdecis[iter]: 
            return 0
        iter += 1
    return 1

def register_link_candidate(self, fitness, cand):
    self.decis[self.inlist] = fitness
    self.linkdecis[self.inlist] = cand
    self.inlist+= 1

def reset_links(self):
    self.prev = PREV_NONE
    self.next = NEXT_NONE
    self.prio = PRIO_DEFAULT
    
def read_path_frame(cor_buf, path_buf, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    filein = None
    linkagein = None
    prioin = None
    targets = -1
    discard = 0

    # File format: first line contains the number of points, then each line is
    # a record of path and correspondence info. We don't need the nuber of points
    # because we read to EOF anyway.

    fname = '{}.{}'.format(corres_file_base, frame_num)
    filein = open(fname, 'r')
    if not filein:
        # Keeping the print until we have proper logging.
        print("Can't open ascii file: {}".format(fname))
        return targets

    read_res = filein.readline()

    if linkage_file_base is not None:
        fname = '{}.{}'.format(linkage_file_base, frame_num)
        linkagein = open(fname, 'r')

        if not linkagein:
            # Keeping the print until we have proper logging.
            print("Can't open linkage file: {}".format(fname))
            return targets

        read_res = linkagein.readline()

    if prio_file_base is not None:
        fname = '{}.{}'.format(prio_file_base, frame_num)
        prioin = open(fname, 'r')

        if not prioin:
            # Keeping the print until we have proper logging.
            print("Can't open prio file: {}".format(fname))
            return targets

        read_res = prioin.readline()

    targets = 0
    for line in filein:
        if linkagein is not None:
            read_res = linkagein.readline()
            items = read_res.split()
            path_buf.prev = int(items[0])
            path_buf.next = int(items[1])
        else:
            # Defaults:
            path_buf.prev = -1
            path_buf.next = -2

        if prioin is not None:
            read_res = prioin.readline()
            items = read_res.split()
            path_buf.prio = int(items[5])
        else:
            path_buf.prio = 4

        # Initialize tracking-related transient variables. These never get saved or restored.
        path_buf.inlist = 0
        path_buf.finaldecis = 1000000.0

        for alt_link in range(POSI):
            path_buf.decis[alt_link] = 0.0
            path_buf.linkdecis[alt_link] = -999

        # Rest of values:
        items = line.split()
        path_buf.x[0] = float(items[1])
        path_buf.x[1] = float(items[2])
        path_buf.x[2] = float(items[3])
        cor_buf.p[0] = int(items[4])
        cor_buf.p[1] = int(items[5])
        cor_buf.p[2] = int(items[6])
        cor_buf.p[3] = int(items[7])
        
        cor_buf.nr = targets + 1

        cor_buf += 1
        path_buf += 1
        
    if filein is not None:
        filein.close()

    if linkagein is not None:
        linkagein.close()

    if prioin is not None:
        prioin.close()

    return targets


def frame_init(new_frame, num_cams, max_targets):
    cam = 0
    new_frame.path_info = [P() for i in range(max_targets)]
    new_frame.correspond = [corres() for i in range(max_targets)]
  
    new_frame.targets = [[] for i in range(num_cams)]
    new_frame.num_targets = [0 for i in range(max_targets)]
    
    while cam < num_cams:
        new_frame.targets[cam] = [target() for i in range(max_targets)]
        new_frame.num_targets[cam] = 0
        cam += 1
    
    new_frame.num_cams = num_cams
    new_frame.max_targets = max_targets
    new_frame.num_parts = 0
    
def free_frame(self):
    # Free path_info and set to None
    free(self.path_info)
    self.path_info = None

    # Free correspond and set to None
    free(self.correspond)
    self.correspond = None

    # Free num_targets and set to None
    free(self.num_targets)
    self.num_targets = None

    # Loop through targets and free each element, then set to None
    for i in range(self.num_cams):
        free(self.targets[self.num_cams - 1])
        self.targets[self.num_cams - 1] = None

    # Free targets and set to None
    free(self.targets)
    self.targets = None
    
        
def write_path_frame(cor_buf, path_buf, num_parts, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    corres_file, linkage_file, prio_file = None, None, None
    corres_fname, linkage_fname, prio_fname = "", "", ""
    pix, success = 0, 0

    corres_fname = "{}.{}".format(corres_file_base, frame_num)
    try:
        corres_file = open(corres_fname, "w")
    except IOError:
        print("Can't open file {} for writing".format(corres_fname))
        return success
    
    linkage_fname = "{}.{}".format(linkage_file_base, frame_num)
    try:
        linkage_file = open(linkage_fname, "w")
    except IOError:
        print("Can't open file {} for writing".format(linkage_fname))
        if corres_file: 
            corres_file.close()
        return success

    if prio_file_base:
        prio_fname = "{}.{}".format(prio_file_base, frame_num)
        try:
            prio_file = open(prio_fname, "w")
        except IOError:
            print("Can't open file {} for writing".format(prio_fname))
            if corres_file: 
                corres_file.close()
            if linkage_file: 
                linkage_file.close()
            return success
        
        prio_file.write('{}\n'.format(num_parts))
        
    linkage_file.write('{}\n'.format(num_parts))
    corres_file.write('{}\n'.format(num_parts))

    for pix in range(num_parts):
        linkage_file.write('{:4d} {:4d} {:10.3f} {:10.3f} {:10.3f}\n'.format(
                path_buf[pix].prev, path_buf[pix].next, path_buf[pix].x[0],
                path_buf[pix].x[1], path_buf[pix].x[2]))
        
        corres_file.write('{:4d} {:9.3f} {:9.3f} {:9.3f} {:4d} {:4d} {:4d} {:4d}\n'.format(
                pix + 1, path_buf[pix].x[0], path_buf[pix].x[1],
                path_buf[pix].x[2], cor_buf[pix].p[0],
                cor_buf[pix].p[1], cor_buf[pix].p[2],
                cor_buf[pix].p[3]))
        
        if not prio_file_base:
            continue
        
        prio_file.write('{:4d} {:4d} {:10.3f} {:10.3f} {:10.3f} {}\n'.format(
                path_buf[pix].prev, path_buf[pix].next, path_buf[pix].x[0],
                path_buf[pix].x[1], path_buf[pix].x[2], path_buf[pix].prio))

    if corres_file: 
        corres_file.close()
    if linkage_file: 
        linkage_file.close()
    if prio_file: 
        prio_file.close()

    success = 1
    return success

def read_frame(self, corres_file_base, linkage_file_base, prio_file_base, target_file_base, frame_num):
    cam = 0
    self.num_parts = read_path_frame(self.correspond, self.path_info, corres_file_base, linkage_file_base, prio_file_base, frame_num)
    
    if (self.num_parts == -1):
        return 0

    if (self.num_targets == 0):
        return 0

    while cam < self.num_cams:
        self.num_targets[cam] = read_targets(self.targets[cam], target_file_base[cam], frame_num)
        
        if (self.num_targets[cam] == -1):
            return 0
        
        cam += 1

    return 1

def write_frame(self, corres_file_base, linkage_file_base,
                prio_file_base, target_file_base, frame_num):
    status = write_path_frame(self.correspond, self.path_info,
                              self.num_parts, corres_file_base, linkage_file_base,
                              prio_file_base, frame_num)
    if status == 0:
        return 0

    for cam in range(self.num_cams):
        status = write_targets(self.targets[cam], self.num_targets[cam],
                               target_file_base[cam], frame_num)
        if status == 0:
            return 0

    return 1


def fb_free(self):
    self._vptr.free(self)

def fb_read_frame_at_end(self, frame_num, read_links):
    return self._vptr.read_frame_at_end(self, frame_num, read_links)

def fb_write_frame_from_start(self, frame_num):
    return self._vptr.write_frame_from_start(self, frame_num)

def fb_base_init(new_buf, buf_len, num_cams, max_targets):
    alloc_frame = None

    new_buf.buf_len = buf_len
    new_buf.num_cams = num_cams

    new_buf._ring_vec = [None] * (buf_len*2)
    new_buf.buf = new_buf._ring_vec + buf_len

    while new_buf.buf != new_buf._ring_vec:
        new_buf.buf -= 1

        alloc_frame = frame()
        frame_init(alloc_frame, num_cams, max_targets)

        new_buf.buf = alloc_frame
        new_buf.buf + buf_len = alloc_frame

    new_buf._vptr = fb_vtable()

def fb_base_free(self):
    self.buf = self._ring_vec

    while self.buf != self._ring_vec + self.buf_len:
        free_frame(self.buf)
        free(self.buf)
        self.buf += 1
    self.buf = None
    
    free(self._ring_vec)
    self._ring_vec = None
    
    free(self._vptr)
    self._vptr = None
    
    
def fb_init(new_buf, buf_len, num_cams, max_targets, corres_file_base, linkage_file_base,
            prio_file_base, target_file_base):
    fb_base_init(new_buf.base, buf_len, num_cams, max_targets)
    
    # Subclass-specific parameters:
    new_buf.corres_file_base = corres_file_base
    new_buf.linkage_file_base = linkage_file_base
    new_buf.prio_file_base = prio_file_base
    new_buf.target_file_base = target_file_base
    
    # Set up the virtual functions table:
    new_buf.base._vptr.free = fb_disk_free
    new_buf.base._vptr.read_frame_at_end = fb_disk_read_frame_at_end
    new_buf.base._vptr.write_frame_from_start = fb_disk_write_frame_from_start

def fb_disk_free(self_base):
    fb_base_free(self_base)

def fb_next(self):
    self.buf += 1
    if self.buf - self._ring_vec >= self.buf_len:
        self.buf = self._ring_vec

def fb_prev(self):
    self.buf -= 1
    if self.buf < self._ring_vec:
        self.buf = self._ring_vec + self.buf_len - 1

def fb_disk_read_frame_at_end(self_base, frame_num, read_links):
    self = self_base
    if read_links:
        return read_frame(self.base.buf[self.base.buf_len - 1], self.corres_file_base,
            self.linkage_file_base, self.prio_file_base, 
            self.target_file_base, frame_num)
    else:
        return read_frame(self.base.buf[self.base.buf_len - 1], self.corres_file_base,
            None, None, self.target_file_base, frame_num)

def fb_disk_write_frame_from_start(self_base, frame_num):
    self = self_base
    return write_frame(self.base.buf[0], self.corres_file_base,
        self.linkage_file_base, self.prio_file_base, self.target_file_base,
        frame_num)
