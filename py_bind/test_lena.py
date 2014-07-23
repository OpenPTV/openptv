from matplotlib.pylab import *
from skimage.data import lena
import numpy as np

from optv.image_processing import py_lowpass_3, py_lowpass_n, py_copy_images
from optv.image_processing import py_filter_3, py_highpass, py_enhance, py_histeq
from optv.image_processing import py_lowpass_3_cb


# use Lena image, but only grayscale (one channel)
# and make it rectangular
a = lena()[:,:,0].astype(np.uint8)
a = a[:300,:].copy()


b = py_lowpass_3(a) 
imshow(np.c_[a,b],cmap='gray')
title("lowpass_3 test")
show()

c = py_lowpass_3_cb(a) 
imshow(np.c_[a,c],cmap='gray')
title("lowpass_3_cb test")
show()


imshow(np.abs(b-c),cmap='gray')
title("lowpass_3 vs 3_cb test")
show()

# imshow(a-b,cmap='gray')
# show()

b = py_lowpass_n(1, a) 
imshow(np.c_[a,b],cmap='gray')
title("lowpass_n test with n = 1")
show()

b = py_lowpass_n(3, a) 
imshow(np.c_[a,b],cmap='gray')
title("lowpass_n test with n = 3")
show()



# test copy_images
b = np.copy(a)
c = py_copy_images(a)
assert all(b == c)


# test filter_3
kernel = [1./9]*9
np.savetxt('filter.par',kernel)

d = a[120:125,130:137].copy(order='C')
# e = np.empty_like(d,order='F')
# imx, imy = d.shape
kernel = np.array([0,1,0,1,2,1,0,1,0])/6.
np.savetxt('filter.par',kernel)
e = py_filter_3(d) 
print "original"
print d
print "filtered"
print e


b = py_filter_3(a) 
imshow(np.c_[a,b],cmap='gray')
title("filter_3 test with 1st filter")
show()

kernel = np.array([1,1,1,1,1,1,1,1,1])/9.
np.savetxt('filter.par',kernel)
e = py_filter_3(d) 
print "original"
print d
print "filtered"
print e

b = py_filter_3(a) 
imshow(np.c_[a,b],cmap='gray')
title("filter_3 test with the 2nd filter")
show()

kernel = np.zeros((9,1))
np.savetxt('filter.par',kernel)
b = py_filter_3(a) 
imshow(np.c_[a,b],cmap='gray')
title("filter_3 test with the corrupted filter")
show()


# test enhance
b = py_enhance(a)
imshow(np.c_[a,b],cmap='gray'); 
title("Enhance test")
show()

# test enhance
b = py_histeq(a)
imshow(np.c_[a,b],cmap='gray'); 
title("Histeq test")
show()

# test highpass
dim_lp = 3; filter_hp = 1
b = py_highpass(a, dim_lp, filter_hp)
imshow(np.c_[a,b],cmap='gray'); 
title("Highpass test")
show()
