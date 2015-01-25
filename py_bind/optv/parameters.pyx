#Implementation of Python binding to parameters.h
from libc.stdlib cimport malloc, free

cdef class Py_mm_np:
      
    def __init__(self, **kwargs):
        if len(kwargs.keys()) == 0:
            self._owns_data = 0
            return
        
        self._owns_data = 1
        self._mm_np = <mm_np *>malloc(sizeof(mm_np))
        
        self._mm_np[0].nlay = kwargs['nlay']
        self._mm_np[0].n1 = kwargs['n1']

        i=0 # setting n2
        while i<len(kwargs['n2']):
            self._mm_np[0].n2[i]=kwargs['n2'][i]
            i=i+1
        i=0 # setting d
        while i<len(kwargs['d']):
            self._mm_np[0].d[i]=kwargs['d'][i]
            i=i+1
            
        self._mm_np[0].n3 = kwargs['n3']
        self._mm_np[0].lut = kwargs['lut']
        
    def toString(self):
        i=0
        n2Str="{"
        while i < sizeof(self._mm_np[0].n2)/sizeof(self._mm_np[0].n2[0]) -1 :
            n2Str = n2Str+ str(self._mm_np[0].n2[i]) + ", "
            i=i+1
        n2Str = n2Str+ str(self._mm_np[0].n2[i]) + "}"
        
        i=0
        dStr="{"
        while i < sizeof(self._mm_np[0].d)/sizeof(self._mm_np[0].d[0]) -1 :
            dStr = dStr+ str(self._mm_np[0].d[i]) + ", "
            i=i+1
        dStr = dStr+ str(self._mm_np[0].d[i]) + "}"
        
        return "nlay=\t{} \nn1=\t{} \nn2=\t{} \nd=\t{} \nn3=\t{} \nlut=\t{} ".format(
                str(self._mm_np[0].nlay),
                str(self._mm_np[0].n1),
                n2Str,
                dStr,
                str(self._mm_np[0].n3),
                str(self._mm_np[0].lut))
        
    def __dealloc__(self):
        if self._owns_data == 0:
            free(self._mm_np)
    
    cdef void set(Py_mm_np self, mm_np * m):
        self._owns_data = 1
        self._mm_np = m
