#Implementation of Python binding to parameters.h
from libc.stdlib cimport malloc, free

cdef class MultimediaParams:

    def __init__(self, **kwargs):
        
        self._mm_np = <mm_np *>malloc(sizeof(mm_np))
        
        self.set_nlay(kwargs['nlay'])
        self.set_n1(kwargs['n1'])
        self.set_n2(kwargs['n2'])
        self.set_d(kwargs['d'])
        self.set_n3(kwargs['n3'])
        self.set_lut(kwargs['lut'])
    
    def get_nlay(self):
        return self._mm_np[0].nlay
    
    def set_nlay(self, nlay):
        self._mm_np[0].nlay = nlay
        
    def get_n1(self):
        return self._mm_np[0].n1
    
    def set_n1(self, n1):
        self._mm_np[0].n1 = n1
        
    def get_n2(self):
        list_size = sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0])
        n2_list = [None] * list_size
        for i in range(list_size):
            n2_list[i] = self._mm_np[0].n2[i]
        return n2_list
    
    def set_n2(self, n2):
        for i in range(len(n2)):
            self._mm_np[0].n2[i] = n2[i]
            
    def get_d(self):
        list_size = sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0])
        d_list = [None] * list_size
        for i in range(list_size):
            d_list[i] = self._mm_np[0].d[i]
        return d_list
        
    def set_d(self, d):
        for i in range(len(d)):
            self._mm_np[0].d[i] = d[i]
        
    def get_n3(self):
        return self._mm_np[0].n3
    
    def set_n3(self, n3):
        self._mm_np[0].n3 = n3
        
    def get_lut(self):
        return self._mm_np[0].lut
    
    def set_lut(self, lut):
        self._mm_np[0].lut = lut
        
    def __str__(self):
        n2_str="{"
        for i in range(sizeof(self._mm_np[0].n2) / sizeof(self._mm_np[0].n2[0]) -1 ):
            n2_str = n2_str+ str(self._mm_np[0].n2[i]) + ", "
        n2_str += str(self._mm_np[0].n2[i+1]) + "}"
        
        d_str="{"
        for i in range(sizeof(self._mm_np[0].d) / sizeof(self._mm_np[0].d[0]) -1 ) :
            d_str += str(self._mm_np[0].d[i]) + ", "
            
        d_str += str(self._mm_np[0].d[i+1]) + "}"
        
        return "nlay=\t{} \nn1=\t{} \nn2=\t{} \nd=\t{} \nn3=\t{} \nlut=\t{} ".format(
                str(self._mm_np[0].nlay),
                str(self._mm_np[0].n1),
                n2_str,
                d_str,
                str(self._mm_np[0].n3),
                str(self._mm_np[0].lut))
        