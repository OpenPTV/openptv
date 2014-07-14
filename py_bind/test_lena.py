from matplotlib.pylab import *
from skimage.data import lena
import numpy as np

# a = np.ascontiguousarray(lena()[:,:,0]).astype(np.uint8)
a = lena()[:,:,0].astype(np.uint8)
b = np.copy(a)

from optv.image_processing import lowpass_3


lowpass_3(a, b)

imshow(np.c_[a,b],cmap='gray')
show()

imshow(a-b,cmap='gray')
show()

a = np.ascontiguousarray(8*np.ones((3,3),dtype=np.uint8))
b = np.copy(a)
lowpass_3(a,b) 
print a
print b
