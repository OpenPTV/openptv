import unittest
from optv.parameters import MultimediaParams

class Test_MultimediaParams(unittest.TestCase):
    
    @staticmethod
    def identical_lists(self, list_1, list_2):
        if len(list_1) != len(list_2):
            return False
        for i in range(len(list_1)):
            if (list_1[i] != list_2[i]):
                return False
        return True
    
    def test_mm_np_instantiation(self):
        
        n2_list = [11,22,33]
        d_list = [55,66,77]
        
        m = MultimediaParams(nlay=3, n1=2, n2=n2_list, d=d_list, n3=4, lut=1)
        
        self.failUnlessEqual(m.get_nlay(), 3)
        self.failUnlessEqual(m.get_n1(), 2)
        self.failUnlessEqual(m.get_n3(), 4)
        self.failUnlessEqual(m.get_lut(), 1)
        
        self.assertTrue(Test_MultimediaParams.identical_lists(self, m.get_d(), d_list))
        self.assertTrue(Test_MultimediaParams.identical_lists(self, m.get_n2(), n2_list))
        
        self.failUnlessEqual(m.__str__(), "nlay=\t3 \nn1=\t2.0 \nn2=\t{11.0, 22.0, 33.0} \nd=\t{55.0, 66.0, 77.0} \nn3=\t4.0 \nlut=\t1 ")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
