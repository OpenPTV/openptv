from matplotlib.pylab import *
from skimage.data import lena
import numpy as np

from optv.image_processing import py_lowpass_3, py_lowpass_n


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



d = a[120:130,130:140]
e = np.zeros_like(d)
imx, imy = d.shape
print imx, imy
py_lowpass_n(4, d, e, imx*imy, imx, imy) 
print e
