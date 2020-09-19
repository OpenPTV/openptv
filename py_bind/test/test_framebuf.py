"""
Check that the bindings do what they are expected to.
The test expects to be run from the py_bind/test/ directory for now,
using the nose test harness [1].

References:
[1] https://nose.readthedocs.org/en/latest/
"""

import unittest, os, numpy as np
from optv.tracking_framebuf import read_targets, Target, TargetArray, Frame

class TestTargets(unittest.TestCase):
    def test_fill_target(self):
        t = Target(pnr=1, tnr=2 ,x=1.5, y=2.5, n=20, nx=4, ny=5, sumg=30)
        self.assertEqual(t.pnr(), 1)
        self.assertEqual(t.tnr(), 2)
        self.assertEqual(t.pos(), (1.5, 2.5))
        self.assertEqual(t.count_pixels(), (20, 4, 5))
        self.assertEqual(t.sum_grey_value(), 30)
        
    def test_fill_target_array(self):
        tarr = TargetArray(2)
        tarr[0].set_pos((1.5, 2.5))
        tarr[1].set_pos((3.5, 4.5))
        
        self.assertEqual(tarr[0].pos(), (1.5, 2.5))
        self.assertEqual(tarr[1].pos(), (3.5, 4.5))

    def test_read_targets(self):
        """Reading a targets file from Python."""
        targs = read_targets("../../liboptv/tests/testing_fodder/sample_", 42)

        self.assertEqual(len(targs), 2)
        self.assertEqual([targ.tnr() for targ in targs], [1, 0])
        self.assertEqual([targ.pos()[0] for targ in targs], [1127., 796.])
        self.assertEqual([targ.pos()[1] for targ in targs], [796., 809.])
    
    def test_sort_y(self):
        """sorting on the Y coordinate in place"""
        targs = read_targets("testing_fodder/frame/cam1.", 333)
        revs = read_targets("testing_fodder/frame/cam1_reversed.", 333)
        revs.sort_y()
        
        for targ, rev in zip(targs, revs):
            self.assertTrue(targ.pos(), rev.pos())
    
    def test_write_targets(self):
        """Round-trip test of writing targets."""
        targs = read_targets("../../liboptv/tests/testing_fodder/sample_", 42)
        targs.write(b"testing_fodder/round_trip.", 1)
        tback = read_targets("testing_fodder/round_trip.", 1)
        
        self.assertEqual(len(targs), len(tback))
        self.assertEqual([targ.tnr() for targ in targs], 
            [targ.tnr() for targ in tback])
        self.assertEqual([targ.pos()[0] for targ in targs], 
            [targ.pos()[0] for targ in tback])
        self.assertEqual([targ.pos()[1] for targ in targs],
            [targ.pos()[1] for targ in tback])
        
    def tearDown(self):
        filename = "testing_fodder/round_trip.0001_targets"
        if os.path.exists(filename):
            os.remove(filename)

class TestFrame(unittest.TestCase):
    def test_read_frame(self):
        """reading a frame"""
        targ_files = ["testing_fodder/frame/cam%d.".encode() % c for c in range(1, 5)]
        frm = Frame(4, corres_file_base=b"testing_fodder/frame/rt_is",
            linkage_file_base=b"testing_fodder/frame/ptv_is", 
            target_file_base=targ_files, frame_num=333)
        
        pos = frm.positions()
        self.assertEqual(pos.shape, (10,3))
        
        targs = frm.target_positions_for_camera(3)
        self.assertEqual(targs.shape, (10,2))
        
        targs_correct = np.array([[ 426., 199.],
            [ 429.,  60.],
            [ 431., 327.],
            [ 509., 315.],
            [ 345., 222.],
            [ 465., 139.],
            [ 487., 403.],
            [ 241., 178.],
            [ 607., 209.],
            [ 563., 238.]])
        np.testing.assert_array_equal(targs, targs_correct)

if __name__ == "__main__":
    unittest.main()

