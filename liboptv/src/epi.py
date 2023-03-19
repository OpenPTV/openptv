from dataclasses import dataclass
import math
from openptv_python.correspondences import MAXCAND
from openptv_python.ray_tracing import ray_tracing
from openptv_python.multimed import move_along_ray
from openptv_python.imgcoord import flat_image_coord
from openptv_python.trafo import correct_brown_affin


@dataclass
class Candidate:
    pnr: int = 0
    tol: int = 0
    corr: int = 0
    
    def __init__(self, pnr=0, tol=0, corr=0):
        self.pnr = pnr
        self.tol = tol
        self.corr = corr

# Define coord_2d struct
class Coord_2d:
    def __init__(self, pnr, x, y):
        self.pnr = pnr
        self.x = x
        self.y = y


def epi_mm(xl, yl, cal1, cal2, mmp, vpar):
    """
    /*  epi_mm() takes a point in images space of one camera, positions of this 
    and another camera and returns the epipolar line (in millimeter units) 
    that corresponds to the point of interest in the another camera space.
    
    Arguments:
    double xl, yl - position of the point on the origin camera's image space,
        in [mm].
    Calibration *cal1 - position of the origin camera
    Calibration *cal2 - position of camera on which the line is projected.
    mm_np *mmp - pointer to multimedia model of the experiment.
    volume_par *vpar - limits the search in 3D for the epipolar line

    Output:
    xmin,ymin and xmax,ymax - end points of the epipolar line in the "second"
        camera 
    */

    Args:
        xl (_type_): _description_
        yl (_type_): _description_
        cal1 (_type_): _description_
        cal2 (_type_): _description_
        mmp (_type_): _description_
        vpar (_type_): _description_

    Returns:
        _type_: _description_
    """
    Zmin, Zmax = 0, 0
    pos, v, X = [0, 0, 0], [0, 0, 0], [0, 0, 0]

    pos, v = ray_tracing(xl, yl, cal1, mmp)

    # calculate min and max depth for position (valid only for one setup)
    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmin_lay[1] - vpar.Zmin_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmax_lay[1] - vpar.Zmax_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    move_along_ray(Zmin, pos, v, X)
    xmin, ymin = flat_image_coord(X, cal2, mmp)

    move_along_ray(Zmax, pos, v, X)
    xmax, ymax = flat_image_coord(X, cal2, mmp)

    return xmin, ymin, xmax, ymax


def epi_mm_2D(xl, yl, cal1, mmp, vpar, out):
    """
    /*  epi_mm_2D() is a very degenerate case of the epipolar geometry use.
    It is valuable only for the case of a single camera with multi-media.
    It takes a point in images space of one (single) camera, positions of this 
    camera and returns the position (in millimeter units) inside the 3D space
    that corresponds to the provided point of interest, limited in the middle of 
    the 3D space, half-way between Zmin and Zmax. In purely 2D experiment, with 
    an infinitely small light sheet thickness or on a flat surface, this will 
    mean the point ray traced through the multi-media into the 3D space.  
    
    Arguments:
    double xl, yl - position of the point in the camera image space [mm].
    Calibration *cal1 - position of the camera
    mm_np *mmp - pointer to multimedia model of the experiment.
    volume_par *vpar - limits the search in 3D for the epipolar line.
    
    Output:
    vec3d out - 3D position of the point in the mid-plane between Zmin and 
        Zmax, which are estimated using volume limits provided in vpar.
*/

    Args:
        xl (_type_): _description_
        yl (_type_): _description_
        cal1 (_type_): _description_
        mmp (_type_): _description_
        vpar (_type_): _description_
        out (_type_): _description_
    """
    pos = [0, 0, 0]
    v = [0, 0, 0]
    Zmin = 0
    Zmax = 0

    pos, v = ray_tracing(xl, yl, cal1, mmp)

    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmin_lay[1] - vpar.Zmin_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])
    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmax_lay[1] - vpar.Zmax_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    move_along_ray(0.5 * (Zmin + Zmax), pos, v, out)


def quality_ratio(a, b):
    return a / b if a < b else b / a


def find_candidate(
    crd, pix, num, xa, ya, xb, yb, n, nx, ny, sumg, cand, vpar, cpar, cal
):
    """
    /*  find_candidate() is searching in the image space of the image all the 
    candidates around the epipolar line originating from another camera. It is 
    a binary search in an x-sorted coord-set, exploits shape information of the
    particles.
    
    Arguments:
    coord_2d *crd - points to an array of detected-points position information.
        the points must be in flat-image (brown/affine corrected) coordinates
        and sorted by their x coordinate, i.e. ``crd[i].x <= crd[i + 1].x``.
    target *pix - array of target information (size, grey value, etc.) 
        structures. pix[j] describes the target corresponding to 
        (crd[...].pnr == j).
    int num - number of particles in the image.
    double xa, xb, ya, yb - end points of the epipolar line [mm].
    int n, nx, ny - total, and per dimension pixel size of a typical target,
        used to evaluate the quality of each candidate by comparing to typical.
    int sumg - same, for the grey value.
    
    Outputs:
    candidate cand[] - array of candidate properties. The .pnr property of cand
        points to an index in the x-sorted corrected detections array 
        (``crd``).
    
    Extra configuration Arguments:
    volume_par *vpar - observed volume dimensions.
    control_par *cpar - general scene data s.a. image size.
    Calibration *cal - position and other parameters on the camera seeing 
        the candidates.
    
    Returns:
    int count - the number of selected candidates, length of cand array. 
        Negative if epipolar line out of sensor array.
*/

    Args:
        crd (_type_): _description_
        pix (_type_): _description_
        num (_type_): _description_
        xa (_type_): _description_
        ya (_type_): _description_
        xb (_type_): _description_
        yb (_type_): _description_
        n (_type_): _description_
        nx (_type_): _description_
        ny (_type_): _description_
        sumg (_type_): _description_
        cand (_type_): _description_
        vpar (_type_): _description_
        cpar (_type_): _description_
        cal (_type_): _description_

    Returns:
        _type_: _description_
    """
    j = 0
    j0 = 0
    dj = 0
    p2 = 0
    count = 0
    tol_band_width = vpar.eps0
    
    # define sensor format for search interrupt
    xmin = (-1) * cpar.pix_x * cpar.imx/2
    xmax = cpar.pix_x * cpar.imx/2
    ymin = (-1) * cpar.pix_y * cpar.imy/2
    ymax = cpar.pix_y * cpar.imy/2
    xmin -= cal.int_par.xh
    ymin -= cal.int_par.yh
    xmax -= cal.int_par.xh
    ymax -= cal.int_par.yh
    correct_brown_affin(xmin, ymin, cal.added_par, xmin, ymin)
    correct_brown_affin(xmax, ymax, cal.added_par, xmax, ymax)
    
    # line equation: y = m*x + b
    if xa == xb: # the line is a point or a vertical line in this camera
        xb += 1e-10 # if we use xa += 1e-10, we always switch later
        
    # equation of a line
    m = (yb - ya) / (xb - xa)
    b = ya - m * xa
    if xa > xb:
        temp = xa
        xa = xb
        xb = temp
        
    if ya > yb:
        temp = ya
        ya = yb
        yb = temp
    
    # If epipolar line out of sensor area, give up.
    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        return -1
    
    # binary search for start point of candidate search
    j0 = num // 2
    dj = num // 4
    while dj > 1:
        if crd[j0].x < xa - tol_band_width:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2
        
    # due to truncation error we might shift to smaller x
    j0 -= 12
    if j0 < 0:
        j0 = 0
        
    # candidate search
    for j in range(j0, num):
        # Since the list is x-sorted, an out of x-bound candidate is after the
        # last possible candidate, so stop.
        if crd[j].x > xb + tol_band_width:
            return count
        
        # Candidate should at the very least be in the epipolar search window
        # to be considred.
        if crd[j].y <= ya - tol_band_width or crd[j].y >= yb + tol_band_width:
            continue
        if crd[j].x <= xa - tol_band_width or crd[j].x >= xb + tol_band_width:
            continue
        
        # Only take candidates within a predefined distance from epipolar line.
        d = abs((crd[j].y - m * crd[j].x - b) / math.sqrt(m * m + 1))
        if d >= tol_band_width:
            continue
        
        p2 = crd[j].pnr
        
        # quality of each parameter is a ratio of the values of the size n, nx, ny and sum of grey values sumg
        qn = quality_ratio(n, pix[p2].n)
        qnx = quality_ratio(nx, pix[p2].nx)
        qny = quality_ratio(ny, pix[p2].ny)
        qsumg = quality_ratio(sumg, pix[p2].sumg)

        # Enforce minimum quality values and maximum candidates
        if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
            continue
        if count >= MAXCAND:
            print(f"More candidates than (maxcand): {count}")
            return count

        # empirical correlation coefficient from shape and brightness parameters
        corr = 4*qsumg + 2*qn + qnx + qny

        # prefer matches with brighter targets
        corr *= float(sumg + pix[p2].sumg)

        cand[count].pnr = j
        cand[count].tol = d
        cand[count].corr = corr
        count += 1

    return count        

