from matplotlib.pylab import *
from skimage.data import lena
import numpy as np

from optv.image_processing import py_lowpass_3, py_lowpass_n, py_copy_images
from optv.image_processing import py_filter_3, py_highpass


a = lena()[:,:,0].astype(np.uint8)
b = py_lowpass_3(a) 
imshow(np.c_[a,b],cmap='gray')
title("lowpass_3 test")
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
a = lena()[:,:,0].astype(np.uint8)
b = np.copy(a)
c = py_copy_images(a)
print all(b == c)


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

kernel = np.array([1,1,1,1,1,1,1,1,1])/9.
np.savetxt('filter.par',kernel)
e = py_filter_3(d) 
print "original"
print d
print "filtered"
print e


# test highpass
a = lena()[:,:,0].astype(np.uint8)
dim_lp = 1; filter_hp = 0
b = py_highpass(a, dim_lp, filter_hp)
imshow(np.c_[a,b],cmap='gray'); 
title("Highpass test")
show()
