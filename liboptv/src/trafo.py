def old_pixel_to_metric(x_metric, y_metric, x_pixel, y_pixel, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode):
    if y_remap_mode == 1:
        y_pixel = 2. * y_pixel + 1.
    elif y_remap_mode == 2:
        y_pixel *= 2.

    x_metric[0] = (x_pixel - float(im_size_x) / 2.) * pix_size_x
    y_metric[0] = (float(im_size_y) / 2. - y_pixel) * pix_size_y


def pixel_to_metric(x_metric, y_metric, x_pixel, y_pixel, parameters):
    old_pixel_to_metric(x_metric, y_metric, x_pixel, y_pixel, parameters.imx, parameters.imy, parameters.pix_x, parameters.pix_y, parameters.chfield)

def old_pixel_to_metric(x_metric, y_metric, x_pixel, y_pixel, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode):
    if y_remap_mode == 1:
        y_pixel = 2.0 * y_pixel + 1.0
    elif y_remap_mode == 2:
        y_pixel *= 2.0
        
    x_metric[0] = (x_pixel - (float(im_size_x) / 2.0)) * pix_size_x
    y_metric[0] = ((float(im_size_y) / 2.0) - y_pixel) * pix_size_y


def old_metric_to_pixel(x_pixel, y_pixel, x_metric, y_metric, im_size_x, im_size_y, pix_size_x, pix_size_y, y_remap_mode):
    x_pixel[0] = x_metric / pix_size_x + im_size_x / 2.0
    y_pixel[0] = im_size_y / 2.0 - y_metric / pix_size_y

    if y_remap_mode == DOUBLED_PLUS_ONE:
        y_pixel[0] = (y_pixel[0] - 1.0) / 2.0
    elif y_remap_mode == DOUBLED:
        y_pixel[0] /= 2.0


def pixel_to_metric(x_metric, y_metric, x_pixel, y_pixel, parameters):
    """
    pixel_to_metric() converts pixel coordinates to metric coordinates, given
    a modern configuration.
    
    Arguments:
    x_metric, y_metric (float): output metric coordinates.
    x_pixel, y_pixel (float): input pixel coordinates.
    parameters (control_par): control structure holding image and pixel sizes.
    y_remap_mode (int): for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
    """
    old_pixel_to_metric(x_metric, y_metric, x_pixel, y_pixel, parameters.imx, 
                        parameters.imy, parameters.pix_x, parameters.pix_y, parameters.chfield)


def old_metric_to_pixel(x_pixel, y_pixel, x_metric, y_metric, im_size_x, 
                        im_size_y, pix_size_x, pix_size_y, y_remap_mode):
    """
    old_metric_to_pixel() converts metric coordinates to pixel coordinates.
    
    Arguments:
    x_pixel, y_pixel (float): input pixel coordinates.
    x_metric, y_metric (float): output metric coordinates.
    im_size_x, im_size_y (int): size in pixels of the corresponding image dimensions.
    pix_size_x, pix_size_y (float): metric size of each pixel on the sensor plane.
    y_remap_mode (int): for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
    """
    x_pixel[0] = (x_metric / pix_size_x) + (im_size_x / 2.)
    y_pixel[0] = (im_size_y / 2.) - (y_metric / pix_size_y)

    if y_remap_mode == 1:
        y_pixel[0] = (y_pixel[0] - 1.) / 2.
    elif y_remap_mode == 2:
        y_pixel[0] /= 2.


def metric_to_pixel(x_pixel, y_pixel, x_metric, y_metric, parameters):
    """
    metric_to_pixel() converts metric coordinates to pixel coordinates, given
    a modern configuration.
    
    Arguments:
    x_pixel, y_pixel (float): input pixel coordinates.
    x_metric, y_metric (float): output metric coordinates.
    parameters (control_par): control structure holding image and pixel sizes.
    y_remap_mode (int): for use with interlaced cameras. Pass 0 for normal use,
        1 for odd lines and 2 for even lines.
    """
    old_metric_to_pixel(x_pixel, y_pixel, x_metric, y_metric, parameters.imx, 
                        parameters.imy, parameters.pix_x, parameters.pix_y, parameters.chfield)


import math

def distort_brown_affin(x, y, ap, x1, y1):
    r = math.sqrt(x*x + y*y)
    if r != 0:
        x += x * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
             + ap.p1 * (r*r + 2*x*x) + 2*ap.p2*x*y
        y += y * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
             + ap.p2 * (r*r + 2*y*y) + 2*ap.p1*x*y
        *x1 = ap.scx * x - math.sin(ap.she) * y
        *y1 = math.cos(ap.she) * y

def correct_brown_affin(x, y, ap, x1, y1):
    correct_brown_affine_exact(x, y, ap, x1, y1, 1e5)



import math

class ap_52:
    def __init__(self, k1, k2, k3, p1, p2, she, scx):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.she = she
        self.scx = scx

def distort_brown_affin(x, y, ap):
    r = math.sqrt(x*x + y*y)
    if r != 0:
        x += x * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
            + ap.p1 * (r*r + 2*x*x) + 2*ap.p2*x*y
        y += y * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
            + ap.p2 * (r*r + 2*y*y) + 2*ap.p1*x*y
        x1 = ap.scx * x - math.sin(ap.she) * y
        y1 = math.cos(ap.she) * y
        return x1, y1
    return x, y

def correct_brown_affin(x, y, ap, tol):
    if x == 0 and y == 0:
        return x, y
    rq = math.sqrt(x*x + y*y)
    xq, yq = x, y
    itnum = 0
    while True:
        r = rq
        xq = (x + yq*math.sin(ap.she)) / ap.scx \
            - xq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
            - ap.p1 * (r*r + 2*xq*xq) - 2*ap.p2*xq*yq
        yq = y/math.cos(ap.she) \
            - yq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
            - ap.p2 * (r*r + 2*yq*yq) - 2*ap.p1*xq*yq
        rq = math.sqrt(xq*xq + yq*yq)
        if rq > 1.2*r:
            rq = 0.5*r
        itnum += 1
        if (abs(rq - r) / r <= tol) or itnum >= 201:
            break
    r = rq
    x1 = (x + yq*math.sin(ap.she)) / ap.scx \
        - xq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
        - ap.p1 * (r*r + 2*xq*xq) - 2*ap.p2*xq*yq
    y1 = y/math.cos(ap.she) \
        - yq * (ap.k1*r*r + ap.k2*r*r*r*r + ap.k3*r*r*r*r*r*r) \
        - ap.p2 * (r*r + 2*yq*yq) - 2*ap.p1*xq*yq
    return x1, y1

import math

def flat_to_dist(flat_x, flat_y, cal):
    # Make coordinates relative to sensor center rather than primary point
    # image coordinates, because distortion formula assumes it, [1] p.180
    flat_x += cal.int_par.xh
    flat_y += cal.int_par.yh
    
    distort_brown_affin(flat_x, flat_y, cal.added_par, dist_x, dist_y)

def dist_to_flat(dist_x, dist_y, cal, tol):
    # Attempt to restore metric flat-image positions from metric real-image coordinates
    # This is an inverse problem so some error is to be expected, but for small enough
    # distortions it's bearable.
    flat_x = dist_x
    flat_y = dist_y
    r = math.sqrt(flat_x**2 + flat_y**2)
    
    itnum = 0
    while True:
        xq = (flat_x + flat_y * math.sin(cal.added_par.she)) / cal.added_par.scx \
             - flat_x * (cal.added_par.k1 * r**2 + cal.added_par.k2 * r**4 + cal.added_par.k3 * r**6) \
             - cal.added_par.p1 * (r**2 + 2*flat_x**2) \
             - 2 * cal.added_par.p2 * flat_x * flat_y
        yq = dist_y / math.cos(cal.added_par.she) \
             - flat_y * (cal.added_par.k1 * r**2 + cal.added_par.k2 * r**4 + cal.added_par.k3 * r**6) \
             - cal.added_par.p2 * (r**2 + 2*flat_y**2) \
             - 2 * cal.added_par.p1 * flat_x * flat_y
        rq = math.sqrt(xq**2 + yq**2)
        
        # Limit divergent iteration
        if rq > 1.2*r:
            rq = 0.5*r
        
        itnum += 1
        
        # Check if we can stop iterating
        if itnum >= 201 or abs(rq - r)/r <= tol:
            break
        
        r = rq
        flat_x = xq
        flat_y = yq
    
    flat_x -= cal.int_par.xh
    flat_y -= cal.int_par.yh
    return flat_x, flat_y

