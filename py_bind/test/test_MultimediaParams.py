import unittest
from optv.parameters import MultimediaParams
import numpy

class Test_MultimediaParams(unittest.TestCase):
    def test_mm_np_instantiation(self):
        
        n2_np = numpy.array([11,22,33])
        d_np = numpy.array([55,66,77])
        
        m = MultimediaParams(nlay=3, n1=2, n2=n2_np, d=d_np, n3=4, lut=1)
        
        self.failUnlessEqual(m.get_nlay(), 3)
        self.failUnlessEqual(m.get_n1(), 2)
        self.failUnlessEqual(m.get_n3(), 4)
        self.failUnlessEqual(m.get_lut(), 1)
        
        numpy.testing.assert_array_equal(m.get_d(), d_np)
        numpy.testing.assert_array_equal(m.get_n2(), n2_np)
        
        self.failUnlessEqual(m.__str__(), "nlay=\t3 \nn1=\t2.0 \nn2=\t{11.0, 22.0, 33.0} \nd=\t{55.0, 66.0, 77.0} \nn3=\t4.0 \nlut=\t1 ")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
