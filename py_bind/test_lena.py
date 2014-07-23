from matplotlib.pylab import *
from skimage.data import lena
import numpy as np

from optv.image_processing import py_lowpass_3, py_lowpass_n, py_copy_images, py_filter_3


a = lena()[:,:,0].astype(np.uint8)
b = np.zeros_like(a)
imx,imy = a.shape
py_lowpass_3(a,b,imx*imy,imx) 

# imshow(np.c_[a,b],cmap='gray')
# show()

# imshow(a-b,cmap='gray')
# show()

# a = lena()[:,:,0].astype(np.uint8)
# b = np.copy(a)
# imx,imy = a.shape
b = np.zeros_like(a)
py_lowpass_n(5, a, b, imx*imy, imx, imy) 

print b[150:155,150:155]	

# imshow(np.c_[a,b],cmap='gray')
# show()

# imshow(a-b,cmap='gray')
# show()



d = a[120:125,130:133]
e = np.zeros_like(d)
imx, imy = d.shape
print imx, imy
py_lowpass_n(1, d, e, imx*imy, imx, imy) 
print e

# test copy_images
a = lena()[:,:,0].astype(np.uint8)
b = np.copy(a)
c = np.empty_like(a)
imx,imy = a.shape
imgsize = imx*imy
py_copy_images(a,c,imgsize)
print all(b == c)


# test filter_3
# a = lena()[:,:,0].astype(np.uint8)
kernel = [1./9]*9
np.savetxt('filter.par',kernel)
# b = py_filter_3(a)
# imshow(np.c_[a,b],cmap='gray'); show()

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