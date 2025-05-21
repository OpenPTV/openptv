import pytest
import numpy as np
from pyoptv.image_processing import (
    filter_3, lowpass_3, fast_box_blur, split, subtract_img, subtract_mask, copy_images, prepare_image
)

def test_filter_3():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    filt = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    result = filter_3(img, filt)
    expected = np.array([[8, 8, 8], [8, 5, 8], [8, 8, 8]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_lowpass_3():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    result = lowpass_3(img)
    expected = np.array([[8, 8, 8], [8, 5, 8], [8, 8, 8]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_fast_box_blur():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    result = fast_box_blur(1, img)
    expected = np.array([[3, 3, 3], [3, 5, 3], [3, 3, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_split():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    result = split(img, 1)
    expected = np.array([[1, 2, 3], [7, 8, 9], [2, 2, 2]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_subtract_img():
    img1 = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=np.uint8)
    img2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    result = subtract_img(img1, img2)
    expected = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_subtract_mask():
    img = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=np.uint8)
    img_mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    result = subtract_mask(img, img_mask)
    expected = np.array([[5, 0, 7], [0, 9, 0], [11, 0, 13]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_copy_images():
    src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    result = copy_images(src)
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

def test_prepare_image():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    dim_lp = 1
    filter_hp = 1
    filter_file = None
    cpar = {'chfield': 0}
    result = prepare_image(img, dim_lp, filter_hp, filter_file, cpar)
    expected = np.array([[3, 3, 3], [3, 5, 3], [3, 3, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)
