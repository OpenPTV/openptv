"""
PyOptv - Pure Python OpenPTV

A pure Python implementation of OpenPTV (Open Source Particle Tracking Velocimetry).
"""

__version__ = "0.1.0"
__author__ = "OpenPTV Contributors"
__email__ = "openptv@googlegroups.com"

# Import main classes and functions
from .calibration import Calibration, Exterior, Interior, Glass, ap_52
from .tracking_frame_buf import FrameBuffer, Target
from .parameters import SequencePar, TrackPar

# Define what gets imported with "from pyoptv import *"
__all__ = [
    "Calibration",
    "Exterior", 
    "Interior",
    "Glass",
    "ap_52",
    "FrameBuffer",
    "Target",
    "SequencePar",
    "TrackPar",
]
