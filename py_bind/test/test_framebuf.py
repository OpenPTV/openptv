"""
Check that the bindings do what they are expected to.
The test expects to be run from the py_bind/test/ directory for now,
using the nose test harness [1].

References:
[1] https://nose.readthedocs.org/en/latest/
"""

import unittest, os
from optv.tracking_framebuf import read_targets, Target, TargetArray

class TestTargets(unittest.TestCase):
    def test_fill_target(self):
        t = Target(pnr=1, tnr=2 ,x=1.5, y=2.5, n=20, nx=4, ny=5, sumg=30)
        self.failUnlessEqual(t.pnr(), 1)
        self.failUnlessEqual(t.tnr(), 2)
        self.failUnlessEqual(t.pos(), (1.5, 2.5))
        self.failUnlessEqual(t.count_pixels(), (20, 4, 5))
        self.failUnlessEqual(t.sum_grey_value(), 30)
        
    def test_fill_target_array(self):
        tarr = TargetArray(2)
        tarr[0].set_pos((1.5, 2.5))
        tarr[1].set_pos((3.5, 4.5))
        
        self.failUnlessEqual(tarr[0].pos(), (1.5, 2.5))
        self.failUnlessEqual(tarr[1].pos(), (3.5, 4.5))

    def test_read_targets(self):
        """Reading a targets file from Python."""
        targs = read_targets("../../liboptv/tests/testing_fodder/sample_", 42)

        self.failUnlessEqual(len(targs), 2)
        self.failUnlessEqual([targ.tnr() for targ in targs], [1, 0])
        self.failUnlessEqual([targ.pos()[0] for targ in targs], [1127., 796.])
        self.failUnlessEqual([targ.pos()[1] for targ in targs], [796., 809.])
    
    def test_write_targets(self):
        """Round-trip test of writing targets."""
        targs = read_targets("../../liboptv/tests/testing_fodder/sample_", 42)
        targs.write("testing_fodder/round_trip.", 1)
        tback = read_targets("testing_fodder/round_trip.", 1)
        
        self.failUnlessEqual(len(targs), len(tback))
        self.failUnlessEqual([targ.tnr() for targ in targs], 
            [targ.tnr() for targ in tback])
        self.failUnlessEqual([targ.pos()[0] for targ in targs], 
            [targ.pos()[0] for targ in tback])
        self.failUnlessEqual([targ.pos()[1] for targ in targs],
            [targ.pos()[1] for targ in tback])
        
    def tearDown(self):
        filename = "testing_fodder/round_trip.0001_targets"
        if os.path.exists(filename):
            os.remove(filename)

