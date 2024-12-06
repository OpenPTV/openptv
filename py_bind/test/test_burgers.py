# -*- coding: utf-8 -*-
"""
Tests for the Tracker with add_particles using Burgers vortex data
with ground truth

Created on Mon Apr 24 10:57:01 2017

@author: alexlib
"""

import unittest
import yaml
import os
import shutil
from optv.tracker import Tracker
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, \
    SequenceParams

framebuf_naming = {
    'corres': b'testing_fodder/burgers/res/rt_is',
    'linkage': b'testing_fodder/burgers/res/ptv_is',
    'prio': b'testing_fodder/burgers/res/whatever'
}


class TestTracker(unittest.TestCase):
    def setUp(self):
        with open(b"testing_fodder/burgers/conf.yaml") as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        seq_cfg = yaml_conf['sequence']

        cals = []
        img_base = []
        print(yaml_conf['cameras'])
        for cix, cam_spec in enumerate(yaml_conf['cameras']):
            cam_spec.setdefault(b'addpar_file', None)
            cal = Calibration()
            cal.from_file(cam_spec['ori_file'].encode(),
                          cam_spec['addpar_file'].encode())
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
        # path = 'testing_fodder/burgers/res'
        # try:
        #     os.mkdir(path)
        # except OSError:
        #     print("Creation of the directory %s failed" % path)
        # else:
        #     print("Successfully created the directory %s " % path)

        shutil.copytree(
           "testing_fodder/burgers/res_orig/", "testing_fodder/burgers/res/")
        shutil.copytree(
           "testing_fodder/burgers/img_orig/", "testing_fodder/burgers/img/")

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward():
            self.assertTrue(self.tracker.current_step() > last_step)
            with open("testing_fodder/burgers/res/rt_is.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                # print(lines)
                if last_step == 10003:
                    self.assertTrue(lines[0] == "4\n")
                else:
                    self.assertTrue(lines[0] == "5\n")
            last_step += 1
        self.tracker.finalize()

    def test_full_forward(self):
        """Automatic full forward tracking run."""
        # os.mkdir('testing_fodder/burgers/res')
        shutil.copytree(
           "testing_fodder/burgers/res_orig/", "testing_fodder/burgers/res/")
        shutil.copytree(
           "testing_fodder/burgers/img_orig/", "testing_fodder/burgers/img/")
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_backward(self):
        """Automatic full backward correction phase."""
        shutil.copytree(
            "testing_fodder/burgers/res_orig/", "testing_fodder/burgers/res/")
        shutil.copytree(
           "testing_fodder/burgers/img_orig/", "testing_fodder/burgers/img/")
        self.tracker.full_forward()
        self.tracker.full_backward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def tearDown(self):
        if os.path.exists("testing_fodder/burgers/res/"):
            shutil.rmtree("testing_fodder/burgers/res/")
        if os.path.exists("testing_fodder/burgers/img/"):
            shutil.rmtree("testing_fodder/burgers/img/")
            # print("there is a /res folder\n")
            # pass


if __name__ == "__main__":
    unittest.main()
