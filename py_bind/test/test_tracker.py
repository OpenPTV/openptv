#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import unittest, yaml, shutil, os
from optv.tracker import Tracker
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, \
    SequenceParams

framebuf_naming = {
    'corres': b'testing_fodder/track/res/particles',
    'linkage': b'testing_fodder/track/res/linkage',
    'prio': b'testing_fodder/track/res/whatever'
}
class TestTracker(unittest.TestCase):
    def setUp(self):
        with open(b"testing_fodder/track/conf.yaml") as f:
            yaml_conf = yaml.load(f)
        seq_cfg = yaml_conf['sequence']
        
        cals = []
        img_base = []
        print(yaml_conf['cameras'])
        for cix, cam_spec in enumerate(yaml_conf['cameras']):
            cam_spec.setdefault(b'addpar_file', None)
            cal = Calibration()
            cal.from_file(cam_spec['ori_file'].encode(), cam_spec['addpar_file'].encode())
            cals.append(cal)
            img_base.append(seq_cfg['targets_template'].format(cam=cix + 1))
  
        cpar = ControlParams(len(yaml_conf['cameras']), **yaml_conf['scene'])
        vpar = VolumeParams(**yaml_conf['correspondences'])
        tpar = TrackingParams(**yaml_conf['tracking'])
        spar = SequenceParams(
            image_base=img_base,
            frame_range=(seq_cfg['first'], seq_cfg['last']))
        
        self.tracker = Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)
        
    def test_forward(self):
        """Manually running a full forward tracking run."""
        shutil.copytree(
            "testing_fodder/track/res_orig/", "testing_fodder/track/res/")
        
        self.tracker.restart()
        last_step = 1
        while self.tracker.step_forward():
            self.failUnless(self.tracker.current_step() > last_step)
            with open("testing_fodder/track/res/linkage.%d" % last_step) as f:
                lines = f.readlines()
                if last_step < 3:
                    self.failUnless(lines[0] == "1\n")
                else:
                    self.failUnless(lines[0] == "2\n")
            last_step += 1
        self.tracker.finalize()
    
    def test_full_forward(self):
        """Automatic full forward tracking run."""
        shutil.copytree(
            "testing_fodder/track/res_orig/", "testing_fodder/track/res/")
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in 
        # the C code.
    
    def test_full_backward(self):
        """Automatic full backward correction phase."""
        shutil.copytree(
            "testing_fodder/track/res_orig/", "testing_fodder/track/res/")
        self.tracker.full_forward()
        self.tracker.full_backward()
        # if it passes without error, we assume it's ok. The actual test is in 
        # the C code.
        
    def tearDown(self):
        if os.path.exists("testing_fodder/track/res/"):
            shutil.rmtree("testing_fodder/track/res/")
        
if __name__ == "__main__":
    unittest.main()
