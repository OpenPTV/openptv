"""
Check that the bindings do what they are expected to.
The test expects to be run from the py_bind/test/ directory for now,
using the nose test harness [1].

References:
[1] https://nose.readthedocs.org/en/latest/
"""

import unittest
from optv.tracking_framebuf import read_targets

class TestTargets(unittest.TestCase):
    def test_read_targets(self):
        """Reading a targets file from Python."""
        targs = read_targets("../../liboptv/tests/testing_fodder/sample_", 42)

        self.failUnlessEqual(len(targs), 2)
        self.failUnlessEqual([targ.tnr() for targ in targs], [1, 0])
        self.failUnlessEqual([targ.pos()[0] for targ in targs], [1127., 796.])
        self.failUnlessEqual([targ.pos()[1] for targ in targs], [796., 809.])

