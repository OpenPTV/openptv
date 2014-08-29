import numpy as np
from optv.intersect import py_intersect

pos1 = np.array([0,0,0],dtype=np.float)
vec1 = np.array([0,0,1],dtype=np.float)
pos2 = np.array([0,0,0],dtype=np.float)
vec2 = np.array([0,0,1],dtype=np.float)

x,y,z = py_intersect(pos1,vec1,pos2,vec2)
print x,y,z


pos1 = np.array([1.0,0.0,0.0])
vec1 = np.array([0.0,-0.707,1.0])
pos2 = np.array([0.0,1.0,0.0])
vec2 = np.array([0.0,0.707,1.0])

x,y,z = py_intersect(pos1,vec1,pos2,vec2)

print x,y,z