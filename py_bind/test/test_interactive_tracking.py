#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Interactive version of the Tracker test, starting as a copy of test_tracker.py
"""

import unittest
import yaml
import shutil
import os
from optv.tracker import Tracker
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, \
    SequenceParams

import matplotlib
matplotlib.use('TkAgg')

framebuf_naming = {
    'corres': b'testing_fodder/track/res/particles',
    'linkage': b'testing_fodder/track/res/linkage',
    'prio': b'testing_fodder/track/res/whatever'
}


class TestInteractiveTracking(unittest.TestCase):
    def setUp(self):
        with open("testing_fodder/track/conf.yaml") as f:
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
        """Manually running a full forward tracking run (interactive version)."""
        import matplotlib.pyplot as plt
        import numpy as np
        # Remove destination if it exists before copying
        if os.path.exists("testing_fodder/track/res/"):
            shutil.rmtree("testing_fodder/track/res/")
        shutil.copytree(
            "testing_fodder/track/res_orig/", "testing_fodder/track/res/")

        self.tracker.restart()
        last_step = 10001
        # Store trajectories for visualization
        traj_x = []
        traj_y = []
        while self.tracker.step_forward():
            # Interactive visualization: plot current targets, choice, and trajectories
            if hasattr(self.tracker, 'framebuf') and hasattr(self.tracker.framebuf, 'get_targets'):
                try:
                    targets = self.tracker.framebuf.get_targets()
                    xs = [t.x for t in targets]
                    ys = [t.y for t in targets]
                    plt.figure(figsize=(8, 6))
                    plt.scatter(xs, ys, c='b', label='Targets')
                    # Visualize the choice (assume first target is chosen for demo)
                    if targets:
                        chosen = targets[0]
                        plt.scatter([chosen.x], [chosen.y], c='r', s=100, marker='*', label='Chosen')
                        traj_x.append(chosen.x)
                        traj_y.append(chosen.y)
                    # Plot trajectory so far
                    if traj_x and traj_y:
                        plt.plot(traj_x, traj_y, 'g--', label='Trajectory')
                    plt.title(f'Frame {self.tracker.current_step()}')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                except Exception as e:
                    print(f"[Interactive] Could not plot targets: {e}")
            print(f"Step: {self.tracker.current_step()} (last_step was {last_step})")
            with open(f"testing_fodder/track/res/linkage.{last_step}") as f:
                lines = f.readlines()
                if last_step == 10003:
                    self.assertTrue(lines[0] == "-1\n")
                else:
                    self.assertTrue(lines[0] == "1\n")
            last_step += 1
            # Pause for user input to step interactively
            input("Press Enter to advance to the next frame...")
        self.tracker.finalize()

    def test_full_forward(self):
        """Automatic full forward tracking run."""
        if os.path.exists("testing_fodder/track/res/"):
            shutil.rmtree("testing_fodder/track/res/")
        shutil.copytree(
            "testing_fodder/track/res_orig/", "testing_fodder/track/res/")
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_backward(self):
        """Automatic full backward correction phase."""
        if os.path.exists("testing_fodder/track/res/"):
            shutil.rmtree("testing_fodder/track/res/")
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
    os.chdir("py_bind/test")
    unittest.main()
