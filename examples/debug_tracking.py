from optv.tracker import Tracker
from optv.track_debugger import TrackingDebugger

def main():
    # Initialize tracker as before
    tracker = Tracker(...)
    
    # Create debugger
    debugger = TrackingDebugger(tracker)
    
    # Track with visualization
    while tracker.step_forward():
        debug_info = tracker.get_debug_info()
        debugger.visualize_step(debug_info)
        
        # Optional: wait for user input to proceed
        input("Press Enter for next step...")

if __name__ == "__main__":
    main()