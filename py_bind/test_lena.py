from matplotlib.pylab import *
from skimage.data import lena
import numpy as np

from optv.image_processing import py_lowpass_3, py_lowpass_n


a = lena()[:,:,0].astype(np.uint8)
b = np.copy(a)
imx,imy = a.shape
py_lowpass_3(a,b,imx*imy,imx) 

imshow(np.c_[a,b],cmap='gray')
show()

imshow(a-b,cmap='gray')
show()

a = 200*np.ones((3,3),dtype=np.uint8)
b = np.copy(a)
imx,imy = a.shape
py_lowpass_3(a,b,imx*imy,imx) 
print a
print b


a = np.ones((3,6),dtype=np.uint8)
b = np.copy(a)
imx,imy = a.shape
py_lowpass_3(a,b, imx*imy, imx) 
print a
print b


a = 200*np.ones((6,3),dtype=np.uint8)
b = np.copy(a)
imx,imy = a.shape
py_lowpass_3(a,b, imx*imy, imx) 
print a
print b



a = lena()[:,:,0].astype(np.uint8)
b = np.copy(a)
imx,imy = a.shape
py_lowpass_n(5, a, b, imx*imy, imx, imy) 	

imshow(np.c_[a,b],cmap='gray')
show()
