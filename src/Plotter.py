import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2


def plot_trajectory_with_time_slider(timestamps, positions, trajectory_label="Trajectory",
                                     gt_timestamps=None, gt_positions=None, gt_label="Ground Truth",
                                     plot_3d=False, orientations=None, gt_orientations=None,
                                     ukf_timestamps=None, ukf_positions=None, ukf_euler=None,
                                     ukf_label="UKF Filtered"):
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
    orientations : array-like, optional, shape (N, 3, 3) or (N, 4)
        Rotation matrices or quaternions [x, y, z, w] for estimated trajectory
    gt_orientations : array-like, optional, shape (N, 3, 3) or (N, 4)
        Rotation matrices or quaternions [x, y, z, w] for ground truth
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    
    # Process orientations if provided
    orientations_available = orientations is not None
    if orientations_available:
        orientations = np.array(orientations)
    
    # Normalize timestamps to start from 0
    t_min = timestamps.min()
    timestamps_norm = timestamps - t_min
    
    # Process ground truth if provided
    gt_available = False
    gt_orientations_available = False
    if gt_timestamps is not None and gt_positions is not None:
        gt_timestamps = np.array(gt_timestamps)
        gt_positions = np.array(gt_positions)
        # Check if ground truth data is non-empty
        if len(gt_timestamps) > 0 and len(gt_positions) > 0:
            gt_timestamps_norm = gt_timestamps - t_min
            gt_available = True
            print(f"Ground truth data: {len(gt_positions)} poses")
            
            # Process GT orientations if provided
            if gt_orientations is not None:
                gt_orientations = np.array(gt_orientations)
                gt_orientations_available = True
        else:
            print("Warning: Ground truth data is empty")
    else:
        print("No ground truth data provided")

    # Process UKF trajectory if provided
    ukf_available = False
    ukf_euler_available = False
    if ukf_timestamps is not None and ukf_positions is not None:
        ukf_positions = np.array(ukf_positions)
        ukf_timestamps = np.array(ukf_timestamps)
        if len(ukf_timestamps) > 0 and len(ukf_positions) > 0:
            ukf_timestamps_norm = ukf_timestamps - t_min
            ukf_available = True
            print(f"UKF trajectory: {len(ukf_positions)} poses")
            if ukf_euler is not None and len(ukf_euler) == len(ukf_timestamps):
                ukf_euler = np.array(ukf_euler)  # (N, 3) [roll, pitch, yaw] radians
                ukf_euler_available = True

    def rotation_matrix_to_euler(R):
        """Convert rotation matrix to roll, pitch, yaw (in degrees)."""
        # Extract roll, pitch, yaw from rotation matrix
        # Assuming ZYX Euler angle convention (yaw-pitch-roll)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.degrees([roll, pitch, yaw])
    
    def quaternion_to_euler(q):
        """Convert quaternion [x, y, z, w] to roll, pitch, yaw (in degrees)."""
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.degrees([roll, pitch, yaw])
    
    # Create figure and axis
    if plot_3d and positions.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Adjust layout to make room for slider and button
    plt.subplots_adjust(bottom=0.15, left=0.15)
    
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
        if ukf_available and ukf_positions.shape[1] >= 3:
            ax.plot(ukf_positions[:, 0], ukf_positions[:, 1], ukf_positions[:, 2],
                   '-', color='orange', alpha=0.3, linewidth=1, label=f'{ukf_label} (full)')
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
        marker, = ax.plot([], [], [], 'ro', markersize=10, label='Current Est. position')
        if gt_available:
            gt_traj_line, = ax.plot([], [], [], 'g-', linewidth=2, label=gt_label)
            gt_marker, = ax.plot([], [], [], 'go', markersize=10, label='Current GT position')
        if ukf_available:
            ukf_traj_line, = ax.plot([], [], [], '-', color='orange', linewidth=2, label=ukf_label)
            ukf_marker, = ax.plot([], [], [], 'o', color='orange', markersize=10, label='Current UKF position')
    else:
        traj_line, = ax.plot([], [], 'b-', linewidth=2, label=trajectory_label)
        marker, = ax.plot([], [], 'ro', markersize=10, label='Current Est. position')
        if gt_available:
            gt_traj_line, = ax.plot([], [], 'g-', linewidth=2, label=gt_label)
            gt_marker, = ax.plot([], [], 'go', markersize=10, label='Current GT position')
        if ukf_available:
            ukf_traj_line, = ax.plot([], [], '-', color='orange', linewidth=2, label=ukf_label)
            ukf_marker, = ax.plot([], [], 'o', color='orange', markersize=10, label='Current UKF position')

    # Create slider
    ax_slider = plt.axes([0.15, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time (s)', 0, timestamps_norm.max(), 
                   valinit=0, valstep=0.01)
    
    # Create play/pause button
    ax_button = plt.axes([0.78, 0.05, 0.08, 0.03])
    button = Button(ax_button, 'Play')
    
    # Animation state
    animation_state = {'is_playing': False, 'timer': None}
    
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
        ukf_pos_3d = None
        ukf_rpy_deg = None
        gt_rpy = None
        est_rpy = None

        # Find index for current time (estimated trajectory)
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

            # Get orientation data if available
            if orientations_available and idx < len(orientations):
                if orientations.shape[-1] == 4:  # Quaternion
                    est_rpy = quaternion_to_euler(orientations[idx])
                elif len(orientations.shape) == 3 and orientations.shape[-2:] == (3, 3):  # Rotation matrix
                    est_rpy = rotation_matrix_to_euler(orientations[idx])
            
            # Update ground truth trajectory if available
            if gt_available:
                gt_idx = np.searchsorted(gt_timestamps_norm, t)
                if gt_idx == 0:
                    gt_idx = 1
                
                gt_traj_line.set_data(gt_positions[:gt_idx, 0], gt_positions[:gt_idx, 1])
                if gt_positions.shape[1] >= 3:
                    gt_traj_line.set_3d_properties(gt_positions[:gt_idx, 2])
                
                # Interpolate current GT position
                if gt_idx < len(gt_timestamps_norm):
                    gt_t_prev = gt_timestamps_norm[gt_idx-1]
                    gt_t_next = gt_timestamps_norm[gt_idx]
                    gt_alpha = (t - gt_t_prev) / (gt_t_next - gt_t_prev + 1e-9)
                    gt_pos = (1 - gt_alpha) * gt_positions[gt_idx-1] + gt_alpha * gt_positions[gt_idx]
                else:
                    gt_pos = gt_positions[-1]
                
                gt_marker.set_data([gt_pos[0]], [gt_pos[1]])
                if gt_positions.shape[1] >= 3:
                    gt_marker.set_3d_properties([gt_pos[2]])
                
                # Get GT orientation data if available
                if gt_orientations_available and gt_idx < len(gt_orientations):
                    if gt_orientations.shape[-1] == 4:  # Quaternion
                        gt_rpy = quaternion_to_euler(gt_orientations[gt_idx])
                    elif len(gt_orientations.shape) == 3 and gt_orientations.shape[-2:] == (3, 3):  # Rotation matrix
                        gt_rpy = rotation_matrix_to_euler(gt_orientations[gt_idx])

            # Update UKF trajectory if available (3D)
            ukf_pos_3d = None
            ukf_rpy_deg = None
            if ukf_available:
                ukf_idx = np.searchsorted(ukf_timestamps_norm, t)
                if ukf_idx == 0:
                    ukf_idx = 1
                ukf_traj_line.set_data(ukf_positions[:ukf_idx, 0], ukf_positions[:ukf_idx, 1])
                ukf_traj_line.set_3d_properties(ukf_positions[:ukf_idx, 2])
                if ukf_idx < len(ukf_timestamps_norm):
                    ua = (t - ukf_timestamps_norm[ukf_idx - 1]) / (
                        ukf_timestamps_norm[ukf_idx] - ukf_timestamps_norm[ukf_idx - 1] + 1e-9)
                    ukf_pos_3d = (1 - ua) * ukf_positions[ukf_idx - 1] + ua * ukf_positions[ukf_idx]
                    if ukf_euler_available:
                        ukf_rpy_rad = (1 - ua) * ukf_euler[ukf_idx - 1] + ua * ukf_euler[ukf_idx]
                        ukf_rpy_deg = np.degrees(ukf_rpy_rad)
                else:
                    ukf_pos_3d = ukf_positions[-1]
                    if ukf_euler_available:
                        ukf_rpy_deg = np.degrees(ukf_euler[-1])
                ukf_marker.set_data([ukf_pos_3d[0]], [ukf_pos_3d[1]])
                ukf_marker.set_3d_properties([ukf_pos_3d[2]])
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

            # Get orientation data if available
            if orientations_available and idx < len(orientations):
                if orientations.shape[-1] == 4:  # Quaternion
                    est_rpy = quaternion_to_euler(orientations[idx])
                elif len(orientations.shape) == 3 and orientations.shape[-2:] == (3, 3):  # Rotation matrix
                    est_rpy = rotation_matrix_to_euler(orientations[idx])
            
            # Update ground truth trajectory if available
            if gt_available:
                gt_idx = np.searchsorted(gt_timestamps_norm, t)
                if gt_idx == 0:
                    gt_idx = 1
                
                gt_traj_line.set_data(gt_positions[:gt_idx, 0], gt_positions[:gt_idx, 1])
                
                # Interpolate current GT position
                if gt_idx < len(gt_timestamps_norm):
                    gt_t_prev = gt_timestamps_norm[gt_idx-1]
                    gt_t_next = gt_timestamps_norm[gt_idx]
                    gt_alpha = (t - gt_t_prev) / (gt_t_next - gt_t_prev + 1e-9)
                    gt_pos = (1 - gt_alpha) * gt_positions[gt_idx-1] + gt_alpha * gt_positions[gt_idx]
                else:
                    gt_pos = gt_positions[-1]
                
                gt_marker.set_data([gt_pos[0]], [gt_pos[1]])

                # Get GT orientation data if available
                if gt_orientations_available and gt_idx < len(gt_orientations):
                    if gt_orientations.shape[-1] == 4:  # Quaternion
                        gt_rpy = quaternion_to_euler(gt_orientations[gt_idx])
                    elif len(gt_orientations.shape) == 3 and gt_orientations.shape[-2:] == (3, 3):  # Rotation matrix
                        gt_rpy = rotation_matrix_to_euler(gt_orientations[gt_idx])

            # Update UKF trajectory if available (2D)
            if ukf_available:
                ukf_idx = np.searchsorted(ukf_timestamps_norm, t)
                if ukf_idx == 0:
                    ukf_idx = 1
                ukf_traj_line.set_data(ukf_positions[:ukf_idx, 0], ukf_positions[:ukf_idx, 1])
                if ukf_idx < len(ukf_timestamps_norm):
                    ua = (t - ukf_timestamps_norm[ukf_idx - 1]) / (ukf_timestamps_norm[ukf_idx] - ukf_timestamps_norm[ukf_idx - 1] + 1e-9)
                    ukf_pos = (1 - ua) * ukf_positions[ukf_idx - 1] + ua * ukf_positions[ukf_idx]
                else:
                    ukf_pos = ukf_positions[-1]
                ukf_marker.set_data([ukf_pos[0]], [ukf_pos[1]])

        # Build text overlay
        text_lines = [f'Time: {t:.2f}s']

        if plot_3d and positions.shape[1] >= 3:
            # UKF block (most prominent when use_vo=False)
            if ukf_pos_3d is not None:
                text_lines.append(f'UKF Pos: [{ukf_pos_3d[0]:.2f}, {ukf_pos_3d[1]:.2f}, {ukf_pos_3d[2]:.2f}]')
            if ukf_rpy_deg is not None:
                text_lines.append(f'UKF RPY: [{ukf_rpy_deg[0]:.1f}°, {ukf_rpy_deg[1]:.1f}°, {ukf_rpy_deg[2]:.1f}°]')

            if gt_available:
                text_lines.append(f'GT  Pos: [{gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}]')
                if gt_rpy is not None:
                    text_lines.append(f'GT  RPY: [{gt_rpy[0]:.1f}°, {gt_rpy[1]:.1f}°, {gt_rpy[2]:.1f}°]')

                if ukf_pos_3d is not None:
                    text_lines.append(f'UKF Pos Err: {np.linalg.norm(ukf_pos_3d - gt_pos):.2f}m')
                if ukf_rpy_deg is not None and gt_rpy is not None:
                    ang_diff = ukf_rpy_deg - gt_rpy
                    ang_diff = (ang_diff + 180) % 360 - 180  # wrap
                    text_lines.append(f'UKF Ang Err: {np.linalg.norm(ang_diff):.1f}°')

                # VO block (only meaningful when use_vo=True)
                text_lines.append(f'Est Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]')
                text_lines.append(f'Est Pos Err: {np.linalg.norm(pos - gt_pos):.2f}m')
                if est_rpy is not None:
                    text_lines.append(f'Est RPY: [{est_rpy[0]:.1f}°, {est_rpy[1]:.1f}°, {est_rpy[2]:.1f}°]')
                    if gt_rpy is not None:
                        ang_e = np.linalg.norm(((est_rpy - gt_rpy) + 180) % 360 - 180)
                        text_lines.append(f'Est Ang Err: {ang_e:.1f}°')
            else:
                text_lines.append(f'Est Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]')
                if est_rpy is not None:
                    text_lines.append(f'Est RPY: [{est_rpy[0]:.1f}°, {est_rpy[1]:.1f}°, {est_rpy[2]:.1f}°]')
        else:
            # 2D branch — keep existing compact display
            if gt_available:
                pos_error = np.linalg.norm(pos[:2] - gt_pos[:2])
                text_lines += [
                    f'Est Pos: [{pos[0]:.2f}, {pos[1]:.2f}]',
                    f'GT  Pos: [{gt_pos[0]:.2f}, {gt_pos[1]:.2f}]',
                    f'Pos Error: {pos_error:.2f}m',
                ]
                if est_rpy is not None:
                    text_lines.append(f'Est RPY: [{est_rpy[0]:.1f}°, {est_rpy[1]:.1f}°, {est_rpy[2]:.1f}°]')
                if gt_rpy is not None:
                    text_lines.append(f'GT  RPY: [{gt_rpy[0]:.1f}°, {gt_rpy[1]:.1f}°, {gt_rpy[2]:.1f}°]')
                if est_rpy is not None and gt_rpy is not None:
                    text_lines.append(f'Ang Error: {np.linalg.norm(est_rpy - gt_rpy):.1f}°')
            else:
                text_lines.append(f'Pos: [{pos[0]:.2f}, {pos[1]:.2f}]')
                if est_rpy is not None:
                    text_lines.append(f'RPY: [{est_rpy[0]:.1f}°, {est_rpy[1]:.1f}°, {est_rpy[2]:.1f}°]')

        time_text.set_text('\n'.join(text_lines))
        
        fig.canvas.draw_idle()
    
    def animate(val):
        """Animate the slider automatically"""
        if animation_state['is_playing']:
            current_val = slider.val
            max_val = timestamps_norm.max()
            step = 0.05  # Time step in seconds
            
            if current_val + step <= max_val:
                slider.set_val(current_val + step)
            else:
                # Loop back to start or stop at end
                slider.set_val(0)
                # Optionally stop at end instead of looping:
                # animation_state['is_playing'] = False
                # button.label.set_text('Play')
        
        return []
    
    def toggle_animation(event):
        """Toggle play/pause state"""
        animation_state['is_playing'] = not animation_state['is_playing']
        
        if animation_state['is_playing']:
            button.label.set_text('Pause')
            # Start animation timer
            if animation_state['timer'] is None:
                animation_state['timer'] = fig.canvas.new_timer(interval=50)  # 50ms = 20 FPS
                animation_state['timer'].add_callback(animate, None)
            animation_state['timer'].start()
        else:
            button.label.set_text('Play')
            # Stop animation timer
            if animation_state['timer'] is not None:
                animation_state['timer'].stop()
    
    # Connect button to toggle function
    button.on_clicked(toggle_animation)
    
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


def display_frame_window(frame, p0=None, p1=None, window_name="Current Frame", 
                        trajectory_info=None, wait_key=1):
    """
    Display the current frame in an OpenCV window with optional feature tracking visualization.
    
    Parameters:
    -----------
    frame : ndarray
        The current frame to display (grayscale or color)
    p0 : array-like, optional
        Previous frame feature points, shape (N, 1, 2) or (N, 2)
    p1 : array-like, optional
        Current frame feature points, shape (N, 1, 2) or (N, 2)
    window_name : str
        Name of the display window
    trajectory_info : dict, optional
        Dictionary with trajectory information to display (e.g., {'position': [x, y, z], 'time': t})
    wait_key : int
        Time in milliseconds to wait for key press (default: 1ms, use 0 to wait indefinitely)
    
    Returns:
    --------
    key : int
        Key code pressed by user (or -1 if no key pressed within wait_key time)
    """
    # Convert grayscale to BGR for visualization
    if len(frame.shape) == 2:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # pylint: disable=no-member
    else:
        display_frame = frame.copy()
    
    # Draw feature points and tracks if provided
    if p0 is not None and p1 is not None:
        # Reshape if needed
        if len(p0.shape) == 3:
            p0 = p0.reshape(-1, 2)
        if len(p1.shape) == 3:
            p1 = p1.reshape(-1, 2)
        
        # Draw tracks (lines from p0 to p1)
        for pt0, pt1 in zip(p0, p1):
            pt0_tuple = tuple(pt0.astype(int))
            pt1_tuple = tuple(pt1.astype(int))
            # Draw line
            cv2.circle(display_frame, pt0_tuple, 1, (0, 255, 255), -1)  # pylint: disable=no-member
            cv2.line(display_frame, pt0_tuple, pt1_tuple, (0, 255, 0), 1)  # pylint: disable=no-member
            # Draw current point
            cv2.circle(display_frame, pt1_tuple, 1, (0, 0, 255), -1)  # pylint: disable=no-member
    
    elif p1 is not None:
        # Only draw current feature points
        if len(p1.shape) == 3:
            p1 = p1.reshape(-1, 2)
        for pt in p1:
            pt_tuple = tuple(pt.astype(int))
            cv2.circle(display_frame, pt_tuple, 3, (255, 255, 0), -1)  # pylint: disable=no-member
    
    # Add trajectory information text overlay
    
    # Display the frame
    display_frame = cv2.resize(display_frame, (0, 0), fx=3, fy=3)  # Resize for better visibility
    cv2.imshow(window_name, display_frame)  # pylint: disable=no-member
    
    # Wait for key press and return key code
    key = cv2.waitKey(wait_key)  # pylint: disable=no-member
    return key


def close_all_windows():
    """Close all OpenCV windows."""
    cv2.destroyAllWindows()  # pylint: disable=no-member


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
