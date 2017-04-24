#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import unittest, yaml
from optv.tracker import Tracker
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, \
    SequenceParams

class TestTracker(unittest.TestCase):
    def setUp(self):
        with open("testing_fodder/track/conf.yaml") as f:
            yaml_conf = yaml.load(f)
        seq_cfg = yaml_conf['sequence']
        
        cals = []
        img_base = []
        for cix, cam_spec in enumerate(yaml_conf['cameras']):
            cam_spec.setdefault('addpar_file', None)
            cal = Calibration()
            cal.from_file(cam_spec['ori_file'], cam_spec['addpar_file'])
            cals.append(cal)
            img_base.append(seq_cfg['targets_template'].format(cam=cix + 1))
            
        cpar = ControlParams(len(yaml_conf['cameras']), **yaml_conf['scene'])
        vpar = VolumeParams(**yaml_conf['correspondences'])
        tpar = TrackingParams(**yaml_conf['tracking'])
        spar = SequenceParams(
            image_base=img_base,
            frame_range=(seq_cfg['first'], seq_cfg['last']))
        
        self.tracker = Tracker(cpar, vpar, tpar, spar, cals)
        
    def test_forward(self):
        """Manually running a full forward tracking run."""
        pass
    
    def test_full_forward(self):
        """Automatic full forward tracking run."""
        pass

if __name__ == "__main__":
    unittest.main()
