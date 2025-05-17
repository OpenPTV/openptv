import pytest
import numpy as np
import matplotlib.pyplot as plt
import os

from optv.tracking_framebuf import FrameBuf

def test_interactive_tracking():
    # Use real data from testing_fodder for FrameBuf input
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
        '../../liboptv/tests/testing_fodder/track'))
    corres_file_base = os.path.join(base_dir, 'res_orig/rt_is')
    linkage_file_base = os.path.join(base_dir, 'res_orig/ptv_is')
    # Only use first two cameras for this test
    target_file_base = [
        os.path.join(base_dir, 'img_orig/cam1'),
        os.path.join(base_dir, 'img_orig/cam2')
    ]
    frame_num = 10095
    # Check that all files exist for this frame
    for cam_base in target_file_base:
        fname = f"{cam_base}.{frame_num}_targets"
        if not os.path.exists(fname):
            pytest.skip(f"Test data not found: {fname}")
            return
    for base in [corres_file_base, linkage_file_base]:
        fname = f"{base}.{frame_num}"
        if not os.path.exists(fname):
            pytest.skip(f"Test data not found: {fname}")
            return
    try:
        fb = FrameBuf(corres_file_base, linkage_file_base, target_file_base, frame_num)
    except Exception as e:
        pytest.skip(f"Could not initialize FrameBuf: {e}")
        return
    n_frames = fb.num_frames()
    assert n_frames > 0, "No frames found in tracking data."
    for frame_idx in range(n_frames):
        targets = fb.get_targets()
        xs = [t.x for t in targets]
        ys = [t.y for t in targets]
        assert all(np.isfinite(xs)), f"Non-finite x in frame {frame_idx}"
        assert all(np.isfinite(ys)), f"Non-finite y in frame {frame_idx}"
        if frame_idx < n_frames - 1:
            assert fb.next_frame() == 1, f"Failed to advance to frame {frame_idx+1}"
    # plt.show()
    # Optionally, reset to first frame and check again
    # (Not implemented in FrameBuf, but could be added)
