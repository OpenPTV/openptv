class Candidate:
    def __init__(self, pnr, tol, corr):
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
    Zmin, Zmax = 0, 0
    pos, v, X = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    
    ray_tracing(xl, yl, cal1, mmp, pos, v)
    
    # calculate min and max depth for position (valid only for one setup)
    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmin_lay[1] - vpar.Zmin_lay[0]) / (vpar.X_lay[1] - vpar.X_lay[0])
    
    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmax_lay[1] - vpar.Zmax_lay[0]) / (vpar.X_lay[1] - vpar.X_lay[0])
    
    move_along_ray(Zmin, pos, v, X)
    flat_image_coord(X, cal2, mmp, xmin, ymin)

    move_along_ray(Zmax, pos, v, X)
    flat_image_coord(X, cal2, mmp, xmax, ymax)

    return xmin, ymin, xmax, ymax

def epi_mm_2D(xl, yl, cal1, mmp, vpar, out):
    pos = [0, 0, 0]
    v = [0, 0, 0]
    Zmin = 0
    Zmax = 0

    ray_tracing(xl, yl, cal1, mmp, pos, v)

    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmin_lay[1] - vpar.Zmin_lay[0]) / (vpar.X_lay[1] - vpar.X_lay[0])
    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (vpar.Zmax_lay[1] - vpar.Zmax_lay[0]) / (vpar.X_lay[1] - vpar.X_lay[0])

    move_along_ray(0.5*(Zmin+Zmax), pos, v, out)

def quality_ratio(a, b):
    return a / b if a < b else b / a


def find_candidate(crd, pix, num, xa, ya, xb, yb, n, nx, ny, sumg, cand, vpar, cpar, cal):
    j = 0
    j0 = 0
    dj = 0
    p2 = 0
    count = 0
    m = 0.0
    b = 0.0
    d = 0.0
    temp = 0.0
    qn = 0.0
    qnx = 0.0
    qny = 0.0
    qsumg = 0.0
    corr = 0.0
    xmin = 0.0
    xmax = 0.0
    ymin = 0.0
    ymax = 0.0
    tol_band_width = vpar.eps0

    xmin = (-1) * cpar.pix_x * cpar.imx / 2
    xmax = cpar.pix_x * cpar.imx / 2
    ymin = (-1) * cpar.pix_y * cpar.imy / 2
    ymax = cpar.pix_y * cpar.imy / 2
    xmin -= cal.int_par.xh
    ymin -= cal.int_par.yh
    xmax -= cal.int_par.xh
    ymax -= cal.int_par.yh

    xmin, ymin = correct_brown_affin(xmin, ymin, cal.added_par)
    xmax, ymax = correct_brown_affin(xmax, ymax, cal.added_par)

    if xa == xb:
        xb += 1e-10

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

    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        return -1

    for dj in [num // 4, num // 8, num // 16, num // 32, num // 64, 1]:
        if crd[j0].x < xa - tol_band_width:
            j0 += dj
        else:
            j0 -= dj

    j0 -= 12

    if j0 < 0:
        j0 = 0

    for j in range(j0, num):
        if crd[j].x > xb + tol_band_width:
            return count

        if crd[j].y <= ya - tol_band_width or crd[j].y >= yb + tol_band_width:
            continue

        if crd[j].x <= xa - tol_band_width or crd[j].x >= xb + tol_band_width:
            continue

        d = fabs((crd[j].y - m * crd[j].x - b) / sqrt(1 + m * m))

        if d > tol_band_width:
            continue

        qn = pix[p1].n
        qnx = pix[p1].nx
        qny = pix[p1].ny
        qsumg = pix[p1].sumg
        sumg = pix[p2].sumg

        count = 0
        for k in range(npix):
            if k == j:
                continue

            # Enforce minimum quality values and maximum candidates
            if (pix[k].n < vpar.cn or pix[k].nx < vpar.cnx or pix[k].ny < vpar.cny or
                pix[k].sumg <= vpar.csumg):
                continue
            if count >= MAXCAND:
                print("More candidates than (maxcand):", count)
                return count

            # Empirical correlation coefficient from shape and brightness parameters
            corr = (4*qsumg + 2*qn + qnx + qny)

            # Prefer matches with brighter targets
            corr *= (sumg + pix[k].sumg)

            cand[count].pnr = j
            cand[count].tol = d
            cand[count].corr = corr
            count += 1

        return count