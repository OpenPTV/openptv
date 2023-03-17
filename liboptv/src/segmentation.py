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
        
def targ_rec(img, targ_par, xmin, xmax, ymin, ymax, cpar, num_cam, pix):
    imx, imy = cpar["imx"], cpar["imy"]
    thres = targ_par["gvthres"][num_cam]
    disco = targ_par["discont"]
    
    # Copy image to a temporary mask
    img0 = img.copy()
    
    # Make sure the min/max coordinates don't cause us to access memory
    # outside the image memory.
    if xmin <= 0: xmin = 1
    if ymin <= 0: ymin = 1
    if xmax >= imx: xmax = imx - 1
    if ymax >= imy: ymax = imy - 1
    
    #  thresholding and connectivity analysis in image  
    targets = []
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            gv = img0[i, j]
            if gv > thres and \
                gv >= img0[i, j-1] and \
                gv >= img0[i, j+1] and \
                gv >= img0[i-1, j] and \
                gv >= img0[i+1, j] and \
                gv >= img0[i-1, j-1] and \
                gv >= img0[i+1, j-1] and \
                gv >= img0[i-1, j+1] and \
                gv >= img0[i+1, j+1]:
                
                # => local maximum, 'peak'
                xn, yn = j, i
                sumg = gv
                img0[i, j] = 0
                xa, xb, ya, yb = xn, xn, yn, yn
                gv -= thres
                x, y = xn * gv, yn * gv
                numpix = 1
                waitlist = np.array([[j, i]])
                
                while len(waitlist) > 0:
                    x, y = np.dot(waitlist[0], [xn, yn])
                    gvref = img[waitlist[0][1], waitlist[0][0]]
                    x4 = np.array([
                        [waitlist[0][0] - 1, waitlist[0][1]], 
                        [waitlist[0][0] + 1, waitlist[0][1]],
                        [waitlist[0][0], waitlist[0][1] - 1],
                        [waitlist[0][0], waitlist[0][1] + 1],
                    ])

                    for n in range(4):
                        xn, yn = x4[n]
                        if not (xn < xmax and yn < ymax):
                            continue
                        gv = img0[yn, xn]

                        # Conditions for threshold, discontinuity, image borders
                        # and peak fitting
                        if gv > thres and \
                            xn > xmin - 1 and xn < xmax + 1 and \
                            yn > ymin - 1 and yn < ymax + 1 and \
                            gv <= gvref+disco and \
                            gvref + disco >= img[yn-1, xn] and \
                            gvref + disco >= img[yn+1, xn] and \
                            gvref + disco >= img[yn, xn-1] and \
                            gvref + disco >= img[yn, xn+1]:
                            
                            # Add to waitlist and mark as "visited"
                            waitlist = np.vstack([waitlist

                            /* compute dot parameters */
                            pix[n_targets].x = x/sumg;
                            pix[n_targets].y = y/sumg;
                            pix[n_targets].grey = sumg;
                            pix[n_targets].ident = n_targets;

                            n_targets++;  /* mark the new dot, reinitialize the counter */
                            n_wait = 0;
                            }
                        }
                    }
                    free(img0);
                    return(n_targets);
                
                
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
