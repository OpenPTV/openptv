def sortgrid(cal, cpar, nfix, fix, num, eps, pix):
    import numpy as np
    
    sorted_pix = np.empty(nfix, dtype=target)
    for i in range(nfix):
        sorted_pix[i].pnr = -999
        
    for i in range(nfix):
        xp, yp = img_coord(fix[i], cal, cpar.mm)
        
        calib_point = metric_to_pixel(xp, yp, cpar)
        
        if (calib_point[0] > -eps) and (calib_point[1] > -eps) and (calib_point[0] < cpar.imx + eps) and (calib_point[1] < cpar.imy + eps):
            j = nearest_neighbour_pix(pix, num, calib_point[0], calib_point[1], eps)
            
            if j != -999:
                sorted_pix[i] = pix[j]
                sorted_pix[i].pnr = i
                
    return sorted_pix


import math

def nearest_neighbour_pix(pix, num, x, y, eps):
    pnr = -999
    dmin = 1e20
    xmin, xmax, ymin, ymax = x-eps, x+eps, y-eps, y+eps
    
    for j in range(num):
        if pix[j].y > ymin and pix[j].y < ymax and pix[j].x > xmin and pix[j].x < xmax:
            d = math.sqrt((x-pix[j].x)*(x-pix[j].x) + (y-pix[j].y)*(y-pix[j].y))
            if d < dmin:
                dmin = d 
                pnr = j
    
    return pnr

def read_sortgrid_par(filename):
    fpp = 0
    eps = 0

    fpp = open(filename, "r")
    if not fpp:
        # handle error
        return None
    if fscanf(fpp, "%d\n", &eps) == 0:
        # handle error
        return None
    fpp.close()

    return eps

def read_calblock(num_points, filename):
    fpp = None
    k = 0
    fix = [0, 0, 0]
    ret = [vec3d(0,0,0)]
    fpp = open(filename, "r")
    if not fpp:
        print("Can't open calibration block file: %s" % filename)
        return None

    while True:
        line = fpp.readline()
        data = line.split()
        if len(data) != 4: # assume end of input
            break
        (dummy, fix[0], fix[1], fix[2]) = data
        ret.append(vec3d(fix[0], fix[1], fix[2]))
        k += 1

    if k == 0:
        print("Empty or badly formatted file: %s" % filename)
        return None

    fpp.close()
    num_points = k
    return ret