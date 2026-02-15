import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def plot_trajectory_with_time_slider(timestamps, positions, trajectory_label="Trajectory", 
                                     gt_timestamps=None, gt_positions=None, gt_label="Ground Truth",
                                     plot_3d=False):
    """
    Create an interactive trajectory plot with a time slider.
    
    Parameters:
    -----------
    timestamps : array-like
        Array of timestamps for the trajectory
    positions : array-like, shape (N, 2) or (N, 3)
        Array of positions [x, y] or [x, y, z] at each timestamp
    trajectory_label : str
        Label for the trajectory line
    gt_timestamps : array-like, optional
        Ground truth timestamps
    gt_positions : array-like, optional
        Ground truth positions
    gt_label : str
        Label for ground truth trajectory
    plot_3d : bool
        If True, create a 3D plot (requires positions to have 3 columns)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    
    # Normalize timestamps to start from 0
    t_min = timestamps.min()
    timestamps_norm = timestamps - t_min
    
    # Process ground truth if provided
    gt_available = False
    if gt_timestamps is not None and gt_positions is not None:
        gt_timestamps = np.array(gt_timestamps)
        gt_positions = np.array(gt_positions)
        # Check if ground truth data is non-empty
        if len(gt_timestamps) > 0 and len(gt_positions) > 0:
            gt_timestamps_norm = gt_timestamps - t_min
            gt_available = True
            print(f"Ground truth data: {len(gt_positions)} poses")
        else:
            print("Warning: Ground truth data is empty")
    else:
        print("No ground truth data provided")
    
    # Create figure and axis
    if plot_3d and positions.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Adjust layout to make room for slider
    plt.subplots_adjust(bottom=0.15)
    
    # Plot full trajectories (faded)
    if plot_3d and positions.shape[1] >= 3:
        full_traj, = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                            'b-', alpha=0.3, linewidth=1, label=f'{trajectory_label} (full)')
        if gt_available and gt_positions.shape[1] >= 3:
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                   'g--', alpha=0.3, linewidth=1, label=f'{gt_label} (full)')
            print(f"Plotted 3D ground truth: X=[{gt_positions[:, 0].min():.3f}, {gt_positions[:, 0].max():.3f}], "
                  f"Y=[{gt_positions[:, 1].min():.3f}, {gt_positions[:, 1].max():.3f}], "
                  f"Z=[{gt_positions[:, 2].min():.3f}, {gt_positions[:, 2].max():.3f}]")
    else:
        full_traj, = ax.plot(positions[:, 0], positions[:, 1], 
                            'b-', alpha=0.3, linewidth=1, label=f'{trajectory_label} (full)')
        if gt_available:
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], 
                   'g--', alpha=0.3, linewidth=1, label=f'{gt_label} (full)')
            print(f"Plotted 2D ground truth: X=[{gt_positions[:, 0].min():.3f}, {gt_positions[:, 0].max():.3f}], "
                  f"Y=[{gt_positions[:, 1].min():.3f}, {gt_positions[:, 1].max():.3f}]")
    
    # Initialize trajectory up to time t and current position marker
    if plot_3d and positions.shape[1] >= 3:
        traj_line, = ax.plot([], [], [], 'b-', linewidth=2, label=trajectory_label)
        marker, = ax.plot([], [], [], 'ro', markersize=10, label='Current position')
    else:
        traj_line, = ax.plot([], [], 'b-', linewidth=2, label=trajectory_label)
        marker, = ax.plot([], [], 'ro', markersize=10, label='Current position')
    
    # Create slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Time (s)', 0, timestamps_norm.max(), 
                   valinit=0, valstep=0.01)
    
    # Text to show current time
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                         verticalalignment='top', fontsize=12, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)) \
               if plot_3d else \
               ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                      verticalalignment='top', fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update(val):
        """Update plot based on slider value"""
        t = slider.val
        
        # Find index for current time
        idx = np.searchsorted(timestamps_norm, t)
        if idx == 0:
            idx = 1
        
        # Update trajectory line up to current time
        if plot_3d and positions.shape[1] >= 3:
            traj_line.set_data(positions[:idx, 0], positions[:idx, 1])
            traj_line.set_3d_properties(positions[:idx, 2])
            
            # Interpolate current position
            if idx < len(timestamps_norm):
                t_prev = timestamps_norm[idx-1]
                t_next = timestamps_norm[idx]
                alpha = (t - t_prev) / (t_next - t_prev + 1e-9)
                pos = (1 - alpha) * positions[idx-1] + alpha * positions[idx]
            else:
                pos = positions[-1]
            
            marker.set_data([pos[0]], [pos[1]])
            marker.set_3d_properties([pos[2]])
        else:
            traj_line.set_data(positions[:idx, 0], positions[:idx, 1])
            
            # Interpolate current position
            if idx < len(timestamps_norm):
                t_prev = timestamps_norm[idx-1]
                t_next = timestamps_norm[idx]
                alpha = (t - t_prev) / (t_next - t_prev + 1e-9)
                pos = (1 - alpha) * positions[idx-1] + alpha * positions[idx]
            else:
                pos = positions[-1]
            
            marker.set_data([pos[0]], [pos[1]])
        
        # Update time text
        time_text.set_text(f'Time: {t:.2f}s\nPosition: [{pos[0]:.2f}, {pos[1]:.2f}' + 
                          (f', {pos[2]:.2f}]' if plot_3d and positions.shape[1] >= 3 else ']'))
        
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update)
    
    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    if plot_3d and positions.shape[1] >= 3:
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory with Time Control')
    else:
        ax.set_title('2D Trajectory with Time Control')
    
    ax.legend()
    ax.grid(True)
    
    if not plot_3d:
        ax.axis('equal')
    
    # Initialize with t=0
    update(0)
    
    plt.show()
    
    return fig, ax

import open3d as o3d
import numpy as np

# =========================
# Sample trajectory (Nx3)
# =========================
points = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.2, 0.1],
    [1.0, 0.4, 0.15],
    [1.5, 0.7, 0.2],
    [2.0, 1.0, 0.25],
    [2.5, 1.3, 0.3],
    [3.0, 1.6, 0.35],
])


# =========================
# Create ground grid
# =========================
def create_grid(size=5, step=1.0):
    points = []
    lines = []
    idx = 0

    for i in np.arange(-size, size + step, step):
        points.append([i, -size, 0])
        points.append([i, size, 0])
        lines.append([idx, idx + 1])
        idx += 2

        points.append([-size, i, 0])
        points.append([size, i, 0])
        lines.append([idx, idx + 1])
        idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]] * len(lines))
    return grid


# =========================
# Create trajectory line
# =========================
def create_trajectory(points):
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(points)
    traj.lines = o3d.utility.Vector2iVector(lines)
    traj.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))
    return traj


# =========================
# Drone marker
# =========================
def create_drone_marker(position, radius=0.08):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(position)
    return sphere


# =========================
# Viewer with time control
# =========================
class TrajectoryViewer:
    def __init__(self, points):
        self.points = points
        self.idx = 0

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Drone Trajectory Viewer")

        self.grid = create_grid()
        self.traj = create_trajectory(points)
        self.marker = create_drone_marker(self.points[self.idx])

        self.vis.add_geometry(self.grid)
        self.vis.add_geometry(self.traj)
        self.vis.add_geometry(self.marker)

        # Key bindings
        self.vis.register_key_callback(262, self.next_frame)  # →
        self.vis.register_key_callback(263, self.prev_frame)  # ←

        print("Controls:")
        print("  →  next time step")
        print("  ←  previous time step")

    def update_marker(self):
        self.vis.remove_geometry(self.marker, reset_bounding_box=False)
        self.marker = create_drone_marker(self.points[self.idx])
        self.vis.add_geometry(self.marker, reset_bounding_box=False)

    def next_frame(self, vis):
        if self.idx < len(self.points) - 1:
            self.idx += 1
            self.update_marker()
        return False

    def prev_frame(self, vis):
        if self.idx > 0:
            self.idx -= 1
            self.update_marker()
        return False

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


# =========================
# Run viewer
# =========================
if __name__ == "__main__":
    # Example 1: Open3D viewer with keyboard controls
    print("Starting Open3D viewer...")
    print("Use arrow keys ← → to navigate through time")
    viewer = TrajectoryViewer(points)
    viewer.run()
    
    # Example 2: Matplotlib viewer with time slider
    print("\nStarting Matplotlib viewer with time slider...")
    
    # Generate example timestamps
    timestamps = np.linspace(0, 10, len(points))  # 10 seconds total
    
    # Create 2D plot
    print("2D trajectory plot:")
    plot_trajectory_with_time_slider(timestamps, points[:, :2], 
                                     trajectory_label="Example Trajectory",
                                     plot_3d=False)
    
    # Create 3D plot
    # print("3D trajectory plot:")
    # plot_trajectory_with_time_slider(timestamps, points, 
    #                                  trajectory_label="Example Trajectory",
    #                                  plot_3d=True)
