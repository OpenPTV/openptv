#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import unittest
import yaml
import shutil
import os
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
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        seq_cfg = yaml_conf['sequence']

        self.cals = []
        img_base = []
        print((yaml_conf['cameras']))
        for cix, cam_spec in enumerate(yaml_conf['cameras']):
            cam_spec.setdefault(b'addpar_file', None)
            cal = Calibration()
            cal.from_file(cam_spec['ori_file'].encode(),
                          cam_spec['addpar_file'].encode())
            self.cals.append(cal)
            img_base.append(seq_cfg['targets_template'].format(cam=cix + 1))

        self.cpar = ControlParams(len(yaml_conf['cameras']), **yaml_conf['scene'])
        self.vpar = VolumeParams(**yaml_conf['correspondences'])
        self.tpar = TrackingParams(**yaml_conf['tracking'])
        self.spar = SequenceParams(
            image_base=img_base,
            frame_range=(seq_cfg['first'], seq_cfg['last']))

        self.tracker = Tracker(self.cpar, self.vpar, self.tpar, self.spar, self.cals, framebuf_naming)

    def test_forward(self):
        """Manually running a full forward tracking run."""
        shutil.copytree(
            "testing_fodder/track/res_orig/", "testing_fodder/track/res/")

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward():
            # print(f"step is {self.tracker.current_step()}\n")
            # print(self.tracker.current_step() > last_step)
            self.assertTrue(self.tracker.current_step() > last_step)
            with open("testing_fodder/track/res/linkage.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                if last_step == 10003:
                    self.assertTrue(lines[0] == "-1\n")
                else:
                    self.assertTrue(lines[0] == "1\n")
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

    def test_tracker_string_handling(self):
        """Test that Tracker handles both strings and bytes correctly"""
        # Using regular strings - will be encoded automatically
        naming_strings = {
            'corres': 'res/rt_is',
            'linkage': 'res/ptv_is',
            'prio': 'res/added'
        }
        tracker1 = Tracker(self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_strings)

        # Using bytes directly - will be passed through
        naming_bytes = {
            'corres': b'res/rt_is',
            'linkage': b'res/ptv_is',
            'prio': b'res/added'
        }
        tracker2 = Tracker(self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_bytes)

        # Using mixed - both will work
        naming_mixed = {
            'corres': 'res/rt_is',  # string
            'linkage': b'res/ptv_is',  # bytes
            'prio': 'res/added'  # string
        }
        tracker3 = Tracker(self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_mixed)

        # Using partial dict - missing keys will use defaults
        naming_partial = {
            'corres': 'res/rt_is'  # only specify what you need to change
        }
        tracker4 = Tracker(self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_partial)

    def tearDown(self):
        if os.path.exists("testing_fodder/track/res/"):
            shutil.rmtree("testing_fodder/track/res/")


if __name__ == "__main__":
    unittest.main()
