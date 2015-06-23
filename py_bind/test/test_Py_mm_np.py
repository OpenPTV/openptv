import unittest
from optv.parameters import Py_mm_np

class Test_Py_mm_np(unittest.TestCase):
    def test_mm_np_instantiation(self):
        
        n2List=[11,22,33]
        dList=[55,66,77]
        
        m=Py_mm_np(nlay=3, n1=2, n2=n2List, d=dList, n3=4, lut=1)
        
        self.assertEqual("nlay=\t3 \nn1=\t2.0 \nn2=\t{11.0, 22.0, 33.0} \nd=\t{55.0, 66.0, 77.0} \nn3=\t4.0 \nlut=\t1 ",m.toString())

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
