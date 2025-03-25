import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class TrackingDebugger:
    def __init__(self, tracker):
        self.tracker = tracker
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize the visualization windows"""
        self.fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(231, projection='3d')
        self.ax_3d.set_title('3D Tracking View')
        
        # Camera view plots
        self.ax_cams = []
        for i in range(4):
            ax = self.fig.add_subplot(234 + i)
            ax.set_title(f'Camera {i+1}')
            self.ax_cams.append(ax)
            
        # Quality factors plot
        self.ax_quality = self.fig.add_subplot(232)
        self.ax_quality.set_title('Candidate Quality Factors')
        
        plt.tight_layout()
        
    def visualize_step(self, debug_info):
        """Visualize one tracking step"""
        self.clear_plots()
        
        # Plot 3D view
        self.plot_3d_scene(debug_info)
        
        # Plot camera views
        self.plot_camera_views(debug_info)
        
        # Plot quality factors
        self.plot_quality_factors(debug_info)
        
        # Add text information
        self.add_info_text(debug_info)
        
        plt.draw()
        plt.pause(0.1)  # Allow for interactive viewing
        
    def plot_3d_scene(self, debug_info):
        """Plot 3D tracking scene"""
        # Plot search volume
        self.plot_search_volume(debug_info.search_bounds)
        
        # Plot candidates
        self.plot_candidates_3d(debug_info.candidates)
        
        # Plot predicted position
        self.ax_3d.scatter(*debug_info.predicted_pos, 
                          color='green', label='Predicted')
        
        # Plot search center
        self.ax_3d.scatter(*debug_info.search_center, 
                          color='red', label='Search Center')
        
    def plot_camera_views(self, debug_info):
        """Plot all camera views"""
        for cam in range(len(self.ax_cams)):
            ax = self.ax_cams[cam]
            
            # Plot search area
            area = debug_info.search_area[cam]
            ax.plot(area[[0,1,2,3,0]], area[[0,1,2,3,0]], 'b-')
            
            # Plot candidates in image space
            for cand in debug_info.candidates:
                if cand.whichcam[cam]:
                    ax.plot(cand.x, cand.y, 'r.')
            
            # Plot search center projection
            ax.plot(*debug_info.search_proj[cam], 'g+')
            
    def plot_quality_factors(self, debug_info):
        """Plot quality factors of candidates"""
        self.ax_quality.bar(range(len(debug_info.quality_factors)), 
                          debug_info.quality_factors)
        self.ax_quality.axvline(debug_info.chosen_candidate, 
                              color='r', linestyle='--')
        
    def add_info_text(self, debug_info):
        """Add textual information about the tracking step"""
        info_text = (
            f"Number of candidates: {debug_info.n_candidates}\n"
            f"Best quality factor: {min(debug_info.quality_factors):.3f}\n"
            f"Search volume size: {np.diff(debug_info.search_bounds)}\n"
            f"Chosen candidate ID: {debug_info.chosen_candidate}"
        )
        self.fig.text(0.02, 0.98, info_text, 
                     verticalalignment='top', 
                     fontfamily='monospace')
        
    def clear_plots(self):
        """Clear all plots for next frame"""
        self.ax_3d.cla()
        for ax in self.ax_cams:
            ax.cla()
        self.ax_quality.cla()