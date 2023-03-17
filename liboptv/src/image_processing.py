filter_t = np.zeros((3, 3), dtype=float)


# Python translation:
def setup_filter_pointers(line):
    ptr = img_lp + (line) + 1
    ptr1 = img
    ptr2 = img + 1
    ptr3 = img + 2

    ptr4 = img + (line)
    ptr5 = ptr4 + 1
    ptr6 = ptr4 + 2

    ptr7 = img + 2*(line)
    ptr8 = ptr7 + 1
    ptr9 = ptr7 + 2

def filter_3(img, img_lp, filt, cpar):
    image_size = cpar.imx * cpar.imy
    sum = 0

    for i in range(3):
        for j in range(3):
            sum += filt[i][j]

    if sum == 0:
        return 0

    end = image_size - cpar.imx - 1

    setup_filter_pointers(cpar.imx)

    for i in range(cpar.imx + 1, end):
        buf = (filt[0][0] * ptr1 + filt[0][1] * ptr2 + filt[0][2] * ptr3 
            + filt[1][0] * ptr4 + filt[1][1] * ptr5 + filt[1][2] * ptr6 
            + filt[2][0] * ptr7 + filt[2][1] * ptr8 + filt[2][2] * ptr9) / sum

        if buf > 255:
            buf = 255
        elif buf < 8: 
            buf = 8

        img_lp[ptr] = int(buf)

    return 1

def lowpass_3(img, img_lp, cpar): 
    ptr = 0 
    ptr1 = 0 
    ptr2 = 0 
    ptr3 = 0 
    ptr4 = 0 
    ptr5 = 0 
    ptr6 = 0 
    ptr7 = 0 
    ptr8 = 0  
    ptr9 = 0  

    end = (cpar.imx * cpar.imy) - cpar.imx - 1

    setup_filter_pointers(cpar.imx)

    for i in range(cpar.imx + 1, end): 
        buf = img[ptr5] + img[ptr1] + img[ptr2] + img[ptr3] + img[ptr4] + img[ptr6] + img[ptr7] + img[ptr8] + img[ptr9] 

        img_lp[ptr] = buf/9

        # increment pointers by one each iteration of the loop  
        ptr += 1  
        ptr1 += 1  
        ptr2 += 1  
        ptr3 += 1  
        ptr4 += 1  
        ptr5 += 1  
        ptr6 += 1  
        ptr7 += 1  
        ptr8 += 1  


def fast_box_blur(filt_span, src, dest, cpar):
    row_accum = [0] * (cpar.imx * cpar.imy)
    col_accum = [0] * cpar.imx
    accum = 0
    n = 2*filt_span + 1
    nq = n*n

    for i in range(cpar.imy):
        row_start = i * cpar.imx

        # first element has no filter around him 
        accum = src[row_start]
        row_accum[row_start] = accum * n

        # Elements 1..filt_span have a growing filter, as much as fits.  Each iteration increases the filter symmetrically, so 2 new elements are taken. 
        for ptr in range(row_start + 1, row_start + 1 + filt_span): 
            ptrl = src[ptr - 2] if ptr - 2 >= 0 else 0  # prevent index out of range error 
            ptrr = src[ptr + 2] if ptr + 2 < len(src) else 0 # prevent index out of range error 

            accum += (ptrl + ptrr)
            m = 3 if ptr == row_start + 1 else (ptr - row_start) * 2  # m is the size of the filter at this point 

            row_accum[ptr] = accum * n / m # So small filters have same weight as the largest size

        # Middle elements, having a constant-size filter. The sum is obtained by adding the coming element and dropping the leaving element, in a sliding window fashion.  
        for ptr in range(row_start + filt_span + 1, row_start + cpar.imx):  
            ptrl = src[ptr - n] if ptr - n >= 0 else 0 # prevent index out of range error  
            ptrr = src[ptr] if ptr < len(src) else 0 # prevent index out of range error  

            accum += (ptrr - ptrl)  
            row_accum[ptr] = accum  

        # last elements in line treated like first ones, mutatis mutandis
        
def split(img: np.ndarray, half_selector: int, cpar: control_par) -> None:
    cond_offs = cpar.imx if half_selector % 2 else 0
    
    if half_selector == 0:
        return
    
    for row in range(cpar.imy // 2):
        for col in range(cpar.imx):
            img[row * cpar.imx + col] = img[2 * row * cpar.imx + cond_offs + col]
    
    img[image_size // 2:] = 2


def subtract_img(img1: np.ndarray, img2: np.ndarray, img_new: np.ndarray):
    """
    Subtract img2 from img1 and store the result in img_new.

    Args:
    img1, img2: numpy arrays containing the original images.
    img_new: numpy array to store the result.
    cpar: control_par object containing image size parameters.
    """
    img_size = img1.size
    for i in range(img_size):
        val = img1[i] - img2[i]
        img_new[i] = 0 if val < 0 else val


def subtract_mask(img, img_mask, img_new, cpar):
    image_size = cpar.imx * cpar.imy
    
    for i in range(image_size):
        if img_mask[i] == 0:
            img_new[i] = 0
        else:
            img_new[i] = img[i]


def copy_images(src: List[int], dest: List[int], cpar: control_par):
    image_size = cpar.imx * cpar.imy

    for i in range(image_size):
        dest[i] = src[i]


import numpy as np
import cv2

def prepare_image(img, img_hp, dim_lp, filter_hp, filter_file, cpar):
    image_size = cpar.imx * cpar.imy
    img_lp = np.zeros(image_size, dtype=np.uint8)
    
    # Apply low-pass filter
    img = img.reshape((cpar.imy, cpar.imx))  # Reshape to 2D image
    img_lp = cv2.boxFilter(img, -1, (dim_lp*2+1, dim_lp*2+1), normalize=True, borderType=cv2.BORDER_CONSTANT)
    
    # Subtract low-pass filtered image from original image
    img_hp = np.subtract(img, img_lp, dtype=np.int16)
    
    # Consider field mode
    if cpar.chfield == 1 or cpar.chfield == 2:
        img_hp = np.array(np.split(img_hp, 2, axis=0)[cpar.chfield-1]).flatten()
        
    # Filter highpass image, if wanted
    if filter_hp == 1:
        img_hp = cv2.boxFilter(img_hp.reshape((cpar.imy, cpar.imx)), -1, (3, 3), normalize=True, borderType=cv2.BORDER_CONSTANT).flatten()
    elif filter_hp == 2:
        try:
            with open(filter_file, "r") as fp:
                filt = np.array(fp.read().split(), dtype=np.float64).reshape((3, 3))
        except Exception:
            return 0

        img_hp = cv2.filter2D(img_hp.reshape((cpar.imy, cpar.imx)), -1, filt, borderType=cv2.BORDER_CONSTANT).flatten()

    return 1

                    