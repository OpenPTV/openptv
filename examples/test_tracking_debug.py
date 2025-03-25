import numpy as np
from optv.tracking_framebuf import read_targets
from optv.parameters import ControlParams, VolumeParams, TrackingParams, SequenceParams
from optv.tracker import Tracker
from optv.track_debugger import TrackingDebugger

def main():
    # Load parameters
    control_params = ControlParams(n_cams=4)
    volume_params = VolumeParams()
    tracking_params = TrackingParams()
    sequence_params = SequenceParams(first=1, last=10)
    
    # Initialize tracker
    tracker = Tracker(
        control_params,
        volume_params,
        tracking_params,
        sequence_params,
        4  # max_links
    )
    
    # Create debugger
    debugger = TrackingDebugger(tracker)
    
    # Track with visualization
    frame = 0
    while tracker.step_forward():
        print(f"Processing frame {frame}")
        debug_info = tracker.get_debug_info()
        debugger.visualize_step(debug_info)
        frame += 1
        
        # Wait for user input (optional)
        input("Press Enter for next frame...")

if __name__ == "__main__":
    main()