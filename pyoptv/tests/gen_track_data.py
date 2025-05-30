"""
Generate a 5-frame trajectory that is pretty degenerates so is good for 
testing. It starts from (0,0,0) and moves in a straight line on the x axis,
at a slow velocity.
"""

import numpy as np
from pyoptv.calibration import Calibration
from pyoptv.parameters import ControlParams
from pyoptv.imgcoord import image_coordinates
from pyoptv.transforms import convert_arr_metric_to_pixel

num_cams = 3
num_frames = 5
velocity = 0.01

part_traject = np.zeros((num_frames,3))
part_traject[:,0] = np.r_[:num_frames]*velocity

# Find targets on each camera.
cpar = ControlParams(3)
cpar.read_control_par("testing_fodder/track/parameters/control_newpart.par")

targs = []
for cam in xrange(num_cams):
    cal = Calibration()
    cal.from_file(
        "testing_fodder/cal/sym_cam%d.tif.ori" % (cam + 1), 
        "testing_fodder/cal/cam1.tif.addpar")
    targs.append(convert_arr_metric_to_pixel(image_coordinates(
        part_traject, cal, cpar.get_multimedia_params()), cpar))

for frame in xrange(num_frames):
    # write 3D positions:
    with open("testing_fodder/track/res_orig/particles.%d" % (frame + 1), "w") as outfile:
        # Note correspondence to the single target in each frame.
        outfile.writelines([
            str(1) + "\n", 
            "{:5d}{:10.3f}{:10.3f}{:10.3f}{:5d}{:5d}{:5d}{:5d}\n".format(
                1, part_traject[frame,0], part_traject[frame,1], 
                part_traject[frame,1], 0, 0, 0, 0)]) 
    
    # write associated targets from all cameras:
    for cam in xrange(num_cams):
        with open("testing_fodder/track/newpart/cam%d.%04d_targets" \
                % (cam + 1, frame + 1), "w") as outfile:
            outfile.writelines([
                str(1) + "\n", 
                "{:5d}{:10.3f}{:10.3f}{:5d}{:5d}{:5d}{:10d}{:5d}\n".format(
                    0, targs[cam][frame, 0], targs[cam][frame, 1], 
                    100, 10, 10, 10000, 0)])

# That's all, folks!
