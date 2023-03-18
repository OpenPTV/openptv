import numpy as np

class Peak:
    def __init__(self):
        self.pos = 0
        self.status = 0
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.n = 0
        self.sumg = 0
        self.x = 0.0
        self.y = 0.0
        self.unr = 0
        self.touch = [0, 0, 0, 0]
        self.n_touch = 0
        
def targ_rec (img, targ_par, xmin, xmax, ymin, ymax, cpar, num_cam, pix):
    n = 0
    n_wait = 0
    n_targets = 0
    sumg = 0
    numpix = 0
    thres = targ_par.gvthres[num_cam]
    disco = targ_par.discont
    
    imx = cpar.imx
    imy = cpar.imy
    
    img0 = [0] * (imx*imy)    # create temporary mask
    img0[:] = img           # copy the original image
    
    if xmin <= 0:
        xmin = 1
    if ymin <= 0:
        ymin = 1
    if xmax >= imx:
        xmax = imx - 1
    if ymax >= imy:
        ymax = imy - 1
    
    waitlist = [[0] * 2 for _ in range(2048)]
    
    xa = 0 
    ya = 0 
    xb = 0 
    yb = 0 
    x4 = [0] * 4 
    y4 = [0] * 4
    
    for i in range(ymin, ymax):  
        for j in range(xmin, xmax):
            gv = img0[i*imx + j]
            if gv > thres:
                if ((gv >= img0[i*imx + j-1])
                    and (gv >= img0[i*imx + j+1])
                    and (gv >= img0[(i-1)*imx + j])
                    and (gv >= img0[(i+1)*imx + j])
                    and (gv >= img0[(i-1)*imx + j-1])
                    and (gv >= img0[(i+1)*imx + j-1])
                    and (gv >= img0[(i-1)*imx + j+1])
                    and (gv >= img0[(i+1)*imx + j+1])):
                    yn = i  
                    xn = j
                    
                    sumg = gv  
                    img0[i*imx + j] = 0
                    
                    xa = xn  
                    xb = xn  
                    ya = yn  
                    yb = yn
                    
                    gv -= thres
                    x = (xn) * gv
                    y = yn * gv
                    numpix = 1
                    waitlist[0][0] = j  
                    waitlist[0][1] = i  
                    n_wait = 1
    

                    while n_wait > 0:
                        gvref = img[imx*(waitlist[0][1]) + (waitlist[0][0])]

                        x4[0] = waitlist[0][0] - 1
                        y4[0] = waitlist[0][1]
                        x4[1] = waitlist[0][0] + 1
                        y4[1] = waitlist[0][1]
                        x4[2] = waitlist[0][0]
                        y4[2] = waitlist[0][1] - 1
                        x4[3] = waitlist[0][0]
                        y4[3] = waitlist[0][1] + 1

                        for n in range(4):
                            xn = x4[n]
                            yn = y4[n]
                            if (xn >= xmax or yn >= ymax or xn < 0 or yn < 0):
                                continue
  
                            gv = img0[imx*yn + xn]

                            if ((gv > thres)
                                and (xn > xmin - 1) and (xn < xmax + 1) 
                                and (yn > ymin - 1) and (yn < ymax + 1)
                                and (gv <= gvref+disco)
                                and (gvref + disco >= img[imx*(yn-1) + xn])
                                and (gvref + disco >= img[imx*(yn+1) + xn])
                                and (gvref + disco >= img[imx*yn + (xn-1)])
                                and (gvref + disco >= img[imx*yn + (xn+1)])):
                                sumg += gv  
                                img0[imx*yn + xn] = 0
                                if (xn < xa):
                                    xa = xn
                                if (xn > xb):
                                    xb = xn
                                if (yn < ya):
                                    ya = yn
                                if (yn > yb):
                                    yb = yn
                                waitlist[n_wait][0] = xn   
                                waitlist[n_wait][1] = yn

                                # Coordinates are weighted by grey value, normed later.
                                x += (xn) * (gv - thres)
                                y += yn * (gv - thres)

                                numpix += 1
                                n_wait += 1

                        n_wait -= 1
                        for m in range(n_wait):
                            waitlist[m][0] = waitlist[m+1][0]
                            waitlist[m][1] = waitlist[m+1][1]
                        waitlist[n_wait][0] = 0  
                        waitlist[n_wait][1] = 0

                    if (xa == (xmin - 1) or ya == (ymin - 1) or 
                        xb == (xmax + 1) or yb == (ymax + 1)): 
                        continue

                    nx = xb - xa + 1  
                    ny = yb - ya + 1

                    if (numpix >= targ_par.nnmin and numpix <= targ_par.nnmax
                        and nx >= targ_par.nxmin and nx <= targ_par.nxmax
                        and ny >= targ_par.nymin and ny <= targ_par.nymax
                        and sumg > targ_par.sumg_min):
                        pix[n_targets].n = numpix
                        pix[n_targets].nx = nx
                        pix[n_targets].ny = ny
                        pix[n_targets].sumg = sumg
                        sumg -= (numpix*thres)
                        # finish the grey-value weighting:
                        x /= sumg  
                        x += 0.5 
                        y /= sumg  
                        y += 0.5
                        pix[n_targets].x = x
                        pix[n_targets].y = y
                        pix[n_targets].tnr = CORRES_NONE
                        pix[n_targets].pnr = n_targets
                        n_targets += 1
                        xn = x  
                        yn = y
            
    return n_targets
                
def check_touch(tpeak, p1, p2):
    done = False
    
    if p2 == 0:
        return
    if p2 == p1:
        return

    # check whether p1, p2 are already marked as touching
    for m in range(tpeak.n_touch):
        if tpeak.touch[m] == p2:
            done = True

    # mark touch event
    if not done:
        tpeak.touch[tpeak.n_touch] = p2
        tpeak.n_touch += 1
        # don't allow for more than 4 touches
        if tpeak.n_touch > 3:
            tpeak.n_touch = 3
