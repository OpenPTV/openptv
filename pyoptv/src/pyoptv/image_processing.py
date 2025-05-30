import numpy as np
from typing import Tuple
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def filter_3(img: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """
    Perform a 3x3 filtering over an image. The first and last lines are not processed at all, the rest uses wrap-around on the image edges. Minimal brightness output in processed pixels is 8.

    Arguments:
    img - original image as a 2D numpy array.
    filt - the 3x3 matrix to apply to the image.

    Returns:
    Filtered image as a 2D numpy array.
    """
    sum_filt = np.sum(filt)
    if sum_filt == 0:
        return img

    img_lp = convolve(img, filt, mode='wrap')
    img_lp = np.clip(img_lp / sum_filt, 8, 255).astype(np.uint8)
    return img_lp

def lowpass_3(img: np.ndarray) -> np.ndarray:
    """
    Perform a 3x3 lowpass filtering over an image.

    Arguments:
    img - original image as a 2D numpy array.

    Returns:
    Filtered image as a 2D numpy array.
    """
    filt = np.ones((3, 3))
    return filter_3(img, filt)
def fast_box_blur(filt_span: int, src: np.ndarray) -> np.ndarray:
    """
    Perform a box blur of an image using a given kernel size.

    Arguments:
    filt_span - how many pixels to take for the average on each side. The equivalent filter kernel is of side 2*filt_size + 1.
    src - source image as a 2D numpy array.

    Returns:
    Blurred image as a 2D numpy array.
    """
    n = 2 * filt_span + 1
    nq = n * n
    dest = np.zeros_like(src)
    row_accum = np.zeros_like(src, dtype=np.int32)
    col_accum = np.zeros(src.shape[1], dtype=np.int32)

    for i in range(src.shape[0]):
        row_start = i * src.shape[1]
        accum = src[i, 0]
        row_accum[i, 0] = accum * n

        for j in range(1, filt_span + 1):
            accum += src[i, j] + src[i, 2 * j]
            row_accum[i, j] = accum * n // (2 * j + 1)

        for j in range(filt_span + 1, src.shape[1] - filt_span):
            accum += src[i, j + filt_span] - src[i, j - filt_span - 1]
            row_accum[i, j] = accum

        for j in range(src.shape[1] - filt_span, src.shape[1]):
            accum -= src[i, 2 * (j - filt_span) - 1] + src[i, 2 * (j - filt_span)]
            row_accum[i, j] = accum * n // (2 * (src.shape[1] - j) - 1)

    for j in range(src.shape[1]):
        col_accum[j] = row_accum[0, j]
        dest[0, j] = col_accum[j] // n

    for i in range(1, filt_span + 1):
        for j in range(src.shape[1]):
            col_accum[j] += row_accum[2 * i - 1, j] + row_accum[2 * i, j]
            dest[i, j] = n * col_accum[j] // nq // (2 * i + 1)

    for i in range(filt_span + 1, src.shape[0] - filt_span):
        for j in range(src.shape[1]):
            col_accum[j] += row_accum[i + filt_span, j] - row_accum[i - filt_span - 1, j]
            dest[i, j] = col_accum[j] // nq

    for i in range(src.shape[0] - filt_span, src.shape[0]):
        for j in range(src.shape[1]):
            col_accum[j] -= row_accum[2 * (i - filt_span) - 1, j] + row_accum[2 * (i - filt_span), j]
            dest[i, j] = n * col_accum[j] // nq // (2 * (src.shape[0] - i) - 1)

    return dest

def split(img: np.ndarray, half_selector: int) -> np.ndarray:
    """
    Cram into the first half of a given image either its even or odd lines. Used with interlaced cameras, a mostly obsolete device. The lower half of the image is set to the number 2.

    Arguments:
    img - the image to modify. Both input and output.
    half_selector - 0 to do nothing, 1 to take odd rows, 2 for even rows

    Returns:
    Modified image as a 2D numpy array.
    """
    if half_selector == 0:
        return img

    img_new = np.copy(img)
    cond_offs = (half_selector % 2) * img.shape[1]

    for row in range(img.shape[0] // 2):
        img_new[row, :] = img[2 * row + cond_offs // img.shape[1], :]

    img_new[img.shape[0] // 2:, :] = 2
    return img_new

def subtract_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Subtract img2 from img1.

    Arguments:
    img1, img2 - original images as 2D numpy arrays.

    Returns:
    Resulting image as a 2D numpy array.
    """
    return np.clip(img1 - img2, 0, None).astype(np.uint8)

def subtract_mask(img: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    """
    Compare img with img_mask and create a masked image img_new. Pixels that are equal to zero in the img_mask are overwritten with a default value (=0) in img_new.

    Arguments:
    img - original image as a 2D numpy array.
    img_mask - mask image as a 2D numpy array.

    Returns:
    Resulting image as a 2D numpy array.
    """
    img_new = np.copy(img)
    img_new[img_mask == 0] = 0
    return img_new

def copy_images(src: np.ndarray) -> np.ndarray:
    """
    Copy one image into another.

    Arguments:
    src - source image as a 2D numpy array.

    Returns:
    Copied image as a 2D numpy array.
    """
    return np.copy(src)

def prepare_image(img: np.ndarray, dim_lp: int, filter_hp: int, filter_file: str, cpar: dict) -> np.ndarray:
    """
    Perform the steps necessary for preparing an image to particle detection: an averaging (smoothing) filter on an image, optionally followed by additional user-defined filter.

    Arguments:
    img - the source image to filter as a 2D numpy array.
    dim_lp - half-width of lowpass filter.
    filter_hp - flag for additional filtering of _hp. 1 for lowpass, 2 for general 3x3 filter given in parameter ``filter_file``.
    filter_file - path to a text file containing the filter matrix to be used in case ``filter_hp == 2``.
    cpar - image details such as size and image half for interlaced cases.

    Returns:
    Filtered image as a 2D numpy array.
    """
    img_lp = fast_box_blur(dim_lp, img)
    img_hp = subtract_img(img, img_lp)

    if cpar['chfield'] == 1 or cpar['chfield'] == 2:
        img_hp = split(img_hp, cpar['chfield'])

    if filter_hp == 1:
        img_hp = lowpass_3(img_hp)
    elif filter_hp == 2:
        with open(filter_file, 'r') as fp:
            filt = np.array([list(map(float, line.split())) for line in fp])
        img_hp = filter_3(img_hp, filt)

    return img_hp

def threshold_image(img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply a binary threshold to the image.

    Arguments:
    img - input image as a 2D numpy array.
    threshold - threshold value.

    Returns:
    Binary image as a 2D numpy array.
    """
    return (img > threshold).astype(np.uint8) * 255

def find_local_maxima(img: np.ndarray) -> np.ndarray:
    """
    Find local maxima in the image.

    Arguments:
    img - input image as a 2D numpy array.

    Returns:
    Image of the same size as img, with local maxima marked.
    """
    from scipy.ndimage import maximum_filter, gaussian_filter

    # Smooth the image with a Gaussian filter
    img_smoothed = gaussian_filter(img, sigma=1)

    # Find maxima using a maximum filter
    img_maxima = maximum_filter(img_smoothed, size=3) == img_smoothed

    return img_maxima.astype(np.uint8) * 255

def label_connected_components(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in the image.

    Arguments:
    img - input image as a 2D numpy array.

    Returns:
    A tuple of (labeled_image, num_features), where labeled_image is the input image with connected components labeled, and num_features is the number of connected components.
    """
    from scipy.ndimage import label

    # Label connected components
    labeled_image, num_features = label(img)

    return labeled_image, num_features
