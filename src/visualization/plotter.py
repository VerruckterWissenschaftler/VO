import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2


def plot_trajectory_with_time_slider(
    timestamps, positions, trajectory_label="Trajectory",
    gt_timestamps=None, gt_positions=None, gt_label="Ground Truth",
    plot_3d=False, orientations=None, gt_orientations=None,
    ukf_timestamps=None, ukf_positions=None, ukf_euler=None, ukf_label="UKF Filtered",
    frames=None, imu_timestamps=None, imu_accel=None,
    velocities=None,
):
    """
    Interactive trajectory plot with time slider.

    Optional side panels:
      - Left: info table with per-frame position, velocity, shift.
      - Right: camera frame nearest to the slider time (pass ``frames`` dict).
      - Bottom centre: velocity subplot (UKF speed + GT speed).
    """
    timestamps = np.array(timestamps)
    positions  = np.array(positions)

    t_min = timestamps.min()
    timestamps_norm = timestamps - t_min

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_available = False
    gt_timestamps_norm = None
    if gt_timestamps is not None and gt_positions is not None:
        gt_timestamps = np.array(gt_timestamps)
        gt_positions  = np.array(gt_positions)
        if len(gt_timestamps) > 0 and len(gt_positions) > 0:
            gt_timestamps_norm = gt_timestamps - t_min
            gt_available = True
            print(f"Ground truth data: {len(gt_positions)} poses")
        else:
            print("Warning: Ground truth data is empty")
    else:
        print("No ground truth data provided")

    # ── UKF ───────────────────────────────────────────────────────────────────
    ukf_available = False
    ukf_timestamps_norm = None
    if ukf_timestamps is not None and ukf_positions is not None:
        ukf_positions  = np.array(ukf_positions)
        ukf_timestamps = np.array(ukf_timestamps)
        if len(ukf_timestamps) > 0 and len(ukf_positions) > 0:
            ukf_timestamps_norm = ukf_timestamps - t_min
            ukf_available = True
            print(f"UKF trajectory: {len(ukf_positions)} poses")

    # ── Velocities ────────────────────────────────────────────────────────────
    velocities_available = velocities is not None and len(velocities) > 0
    if velocities_available:
        velocities = np.array(velocities)

    # ── GT velocity (finite difference) ───────────────────────────────────────
    gt_vel_available = False
    gt_vel_times_norm = gt_vel_speed = None
    if gt_available and len(gt_timestamps_norm) > 1:
        dt_gt = np.diff(gt_timestamps_norm)
        dt_gt = np.maximum(dt_gt, 1e-9)
        gt_vel_xyz    = np.diff(gt_positions, axis=0) / dt_gt[:, None]
        gt_vel_times_norm = gt_timestamps_norm[1:]
        gt_vel_speed  = np.linalg.norm(gt_vel_xyz, axis=1)
        gt_vel_available = True

    # ── UKF speed (for velocity subplot) ─────────────────────────────────────
    ukf_speed_available = velocities_available and ukf_available
    ukf_speed = np.linalg.norm(velocities, axis=1) if ukf_speed_available else None

    # ── Frames ────────────────────────────────────────────────────────────────
    frames_available = False
    frame_times = None
    if frames is not None and len(frames) > 0:
        frame_times      = np.array(sorted(frames.keys()))
        frames_available = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _interp(ts_norm, arr, t):
        """Interpolate arr at normalised time t; return (value, index)."""
        n = len(ts_norm)
        if n == 0:
            return np.zeros(arr.shape[1] if arr.ndim > 1 else 1), 0
        if n == 1:
            return arr[0].copy(), 0
        idx = int(np.clip(np.searchsorted(ts_norm, t), 1, n - 1))
        alpha = np.clip(
            (t - ts_norm[idx - 1]) / (ts_norm[idx] - ts_norm[idx - 1] + 1e-9),
            0.0, 1.0,
        )
        return (1.0 - alpha) * arr[idx - 1] + alpha * arr[idx], idx

    # ── Figure layout ─────────────────────────────────────────────────────────
    is_3d = plot_3d and positions.shape[1] >= 3

    if frames_available:
        fig = plt.figure(figsize=(20, 11))
        l_info, w_info = 0.01, 0.17
        l_traj, w_traj = 0.20, 0.43
        l_frm,  w_frm  = 0.65, 0.22
        prev_rect   = [l_traj,        0.04, 0.04, 0.03]
        slider_rect = [l_traj + 0.05, 0.04, 0.32, 0.03]
        next_rect   = [l_traj + 0.38, 0.04, 0.04, 0.03]
        button_rect = [l_frm,         0.04, 0.08, 0.03]
    else:
        fig = plt.figure(figsize=(16, 11))
        l_info, w_info = 0.01, 0.20
        l_traj, w_traj = 0.23, 0.73
        prev_rect   = [l_traj,        0.04, 0.04, 0.03]
        slider_rect = [l_traj + 0.05, 0.04, 0.38, 0.03]
        next_rect   = [l_traj + 0.44, 0.04, 0.04, 0.03]
        button_rect = [l_traj + 0.49, 0.04, 0.08, 0.03]

    panel_rect = [l_info,  0.11, w_info, 0.85]
    traj_rect  = [l_traj,  0.40, w_traj, 0.56]
    vel_rect   = [l_traj,  0.11, w_traj, 0.24]

    # Info panel (left)
    ax_info = fig.add_axes(panel_rect)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for sp in ax_info.spines.values():
        sp.set_linewidth(0.8)
    info_text_obj = ax_info.text(
        0.05, 0.98, '',
        transform=ax_info.transAxes,
        verticalalignment='top',
        fontsize=7.5,
        fontfamily='monospace',
    )

    # Trajectory axes (centre top)
    ax = fig.add_axes(traj_rect, projection='3d') if is_3d else fig.add_axes(traj_rect)

    # Velocity subplot (centre bottom)
    ax_vel = fig.add_axes(vel_rect)
    ax_vel.set_xlabel('time (s)', fontsize=8)
    ax_vel.set_ylabel('speed (m/s)', fontsize=8)
    ax_vel.set_title('Speed', fontsize=9, fontweight='bold')
    ax_vel.tick_params(labelsize=7)
    ax_vel.grid(True, linewidth=0.4)

    t_max = timestamps_norm.max()

    if ukf_speed_available:
        ax_vel.plot(ukf_timestamps_norm, ukf_speed,
                    color='tab:orange', linewidth=1.0, alpha=0.8, label='UKF speed')
    if gt_vel_available:
        ax_vel.plot(gt_vel_times_norm, gt_vel_speed,
                    color='tab:green', linewidth=1.0, alpha=0.8, linestyle='--', label='GT speed')
    if ukf_speed_available or gt_vel_available:
        ax_vel.legend(fontsize=7, loc='upper right')

    vel_vline = ax_vel.axvline(0, color='red', linewidth=1.2, alpha=0.9)

    # Frame panel (right, optional)
    _FRAME_W, _FRAME_H = 400, 300

    frame_img = None
    if frames_available:
        ax_frame = fig.add_axes([l_frm, 0.11, w_frm, 0.85])
        ax_frame.axis('off')
        ax_frame.set_title('Camera Frame', fontsize=9)
        first = cv2.resize(frames[frame_times[0]], (_FRAME_W, _FRAME_H))  # pylint: disable=no-member
        cmap  = 'gray' if first.ndim == 2 else None
        frame_img = ax_frame.imshow(first, cmap=cmap, aspect='equal')

    # ── Plot full (faded) trajectories ────────────────────────────────────────
    if is_3d:
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', alpha=0.3, linewidth=1, label=f'{trajectory_label} (full)')
        if gt_available and gt_positions.shape[1] >= 3:
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                    'g--', alpha=0.3, linewidth=1, label=f'{gt_label} (full)')
        if ukf_available and ukf_positions.shape[1] >= 3:
            ax.plot(ukf_positions[:, 0], ukf_positions[:, 1], ukf_positions[:, 2],
                    '-', color='orange', alpha=0.3, linewidth=1, label=f'{ukf_label} (full)')

        traj_line, = ax.plot([], [], [], 'b-', linewidth=2, label=trajectory_label)
        marker,    = ax.plot([], [], [], 'ro', markersize=8, label='Current Est.')
        if gt_available:
            gt_traj_line, = ax.plot([], [], [], 'g-', linewidth=2, label=gt_label)
            gt_marker,    = ax.plot([], [], [], 'go', markersize=8, label='Current GT')
        if ukf_available:
            ukf_traj_line, = ax.plot([], [], [], '-', color='orange', linewidth=2, label=ukf_label)
            ukf_marker,    = ax.plot([], [], [], 'o', color='orange', markersize=8, label='Current UKF')
    else:
        ax.plot(positions[:, 0], positions[:, 1],
                'b-', alpha=0.3, linewidth=1, label=f'{trajectory_label} (full)')
        if gt_available:
            ax.plot(gt_positions[:, 0], gt_positions[:, 1],
                    'g--', alpha=0.3, linewidth=1, label=f'{gt_label} (full)')
        if ukf_available:
            ax.plot(ukf_positions[:, 0], ukf_positions[:, 1],
                    '-', color='orange', alpha=0.3, linewidth=1, label=f'{ukf_label} (full)')

        traj_line, = ax.plot([], [], 'b-', linewidth=2, label=trajectory_label)
        marker,    = ax.plot([], [], 'ro', markersize=8, label='Current Est.')
        if gt_available:
            gt_traj_line, = ax.plot([], [], 'g-', linewidth=2, label=gt_label)
            gt_marker,    = ax.plot([], [], 'go', markersize=8, label='Current GT')
        if ukf_available:
            ukf_traj_line, = ax.plot([], [], '-', color='orange', linewidth=2, label=ukf_label)
            ukf_marker,    = ax.plot([], [], 'o', color='orange', markersize=8, label='Current UKF')

    # ── Slider and button ─────────────────────────────────────────────────────
    ax_prev  = fig.add_axes(prev_rect)
    btn_prev = Button(ax_prev, '◀')

    ax_slider = fig.add_axes(slider_rect)
    slider = Slider(ax_slider, 'Time (s)', 0, t_max, valinit=0, valstep=0.01)

    ax_next  = fig.add_axes(next_rect)
    btn_next = Button(ax_next, '▶')

    ax_button = fig.add_axes(button_rect)
    button = Button(ax_button, 'Play')

    animation_state = {'is_playing': False, 'timer': None}

    # ── Update callback ───────────────────────────────────────────────────────

    def update(val):
        t     = slider.val
        t_abs = t + t_min

        # Primary trajectory
        pos, idx  = _interp(timestamps_norm, positions, t)
        vo_shift  = positions[idx] - positions[idx - 1] if idx > 0 else np.zeros(3)
        vel       = velocities[idx] if velocities_available and idx < len(velocities) else None

        # GT
        gt_pos = gt_shift = None
        gt_idx = 0
        if gt_available:
            gt_pos, gt_idx = _interp(gt_timestamps_norm, gt_positions, t)
            gt_shift = gt_positions[gt_idx] - gt_positions[gt_idx - 1] if gt_idx > 0 else np.zeros(3)

        # GT velocity at current time
        gt_vel_cur = None
        if gt_vel_available:
            gi = int(np.clip(np.searchsorted(gt_vel_times_norm, t), 0, len(gt_vel_times_norm) - 1))
            gt_vel_cur = float(gt_vel_speed[gi])

        # UKF
        ukf_pos = ukf_shift = None
        ukf_idx = 0
        if ukf_available:
            ukf_pos, ukf_idx = _interp(ukf_timestamps_norm, ukf_positions, t)
            ukf_shift = ukf_positions[ukf_idx] - ukf_positions[ukf_idx - 1] if ukf_idx > 0 else np.zeros(3)

        # ── Update trajectory lines ───────────────────────────────────────────
        if is_3d:
            traj_line.set_data(positions[:idx, 0], positions[:idx, 1])
            traj_line.set_3d_properties(positions[:idx, 2])
            marker.set_data([pos[0]], [pos[1]])
            marker.set_3d_properties([pos[2]])
            if gt_available:
                gt_traj_line.set_data(gt_positions[:gt_idx, 0], gt_positions[:gt_idx, 1])
                gt_traj_line.set_3d_properties(
                    gt_positions[:gt_idx, 2] if gt_positions.shape[1] >= 3 else np.zeros(gt_idx))
                gt_marker.set_data([gt_pos[0]], [gt_pos[1]])
                gt_marker.set_3d_properties([gt_pos[2]])
            if ukf_available:
                ukf_traj_line.set_data(ukf_positions[:ukf_idx, 0], ukf_positions[:ukf_idx, 1])
                ukf_traj_line.set_3d_properties(ukf_positions[:ukf_idx, 2])
                ukf_marker.set_data([ukf_pos[0]], [ukf_pos[1]])
                ukf_marker.set_3d_properties([ukf_pos[2]])
        else:
            traj_line.set_data(positions[:idx, 0], positions[:idx, 1])
            marker.set_data([pos[0]], [pos[1]])
            if gt_available:
                gt_traj_line.set_data(gt_positions[:gt_idx, 0], gt_positions[:gt_idx, 1])
                gt_marker.set_data([gt_pos[0]], [gt_pos[1]])
            if ukf_available:
                ukf_traj_line.set_data(ukf_positions[:ukf_idx, 0], ukf_positions[:ukf_idx, 1])
                ukf_marker.set_data([ukf_pos[0]], [ukf_pos[1]])

        # ── Update velocity vline ─────────────────────────────────────────────
        vel_vline.set_xdata([t, t])

        # ── Update frame panel ────────────────────────────────────────────────
        if frames_available and frame_img is not None:
            fi = int(np.clip(np.searchsorted(frame_times, t_abs), 0, len(frame_times) - 1))
            frame_img.set_data(cv2.resize(frames[frame_times[fi]], (_FRAME_W, _FRAME_H)))  # pylint: disable=no-member

        # ── Build info text ───────────────────────────────────────────────────
        def f3(v):
            return f'[{v[0]:+6.2f},{v[1]:+6.2f},{v[2]:+6.2f}]'

        def f3s(v):
            return f'[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]'

        traj_lbl = trajectory_label[:18].ljust(18)
        lines = [f't = {t:.2f} s', '']

        lines += [f'─ {traj_lbl} ─']
        lines += [f'Pos  {f3(pos)}']
        lines += [f'Vel  {f3s(vel)}' if vel is not None else 'Vel  N/A']
        lines += [f'Shft {f3s(vo_shift)}']
        lines += ['']

        if gt_pos is not None:
            lines += ['─ GT ─────────────────']
            lines += [f'Pos  {f3(gt_pos)}']
            lines += [f'Vel  |v|={gt_vel_cur:.3f} m/s' if gt_vel_cur is not None else 'Vel  N/A']
            lines += [f'Shft {f3s(gt_shift)}']
            traj_err = np.linalg.norm(pos[:3] - gt_pos[:3])
            lines += [f'ErrP {traj_err:.3f} m']
            lines += ['']

        if ukf_pos is not None:
            lines += ['─ UKF ────────────────']
            lines += [f'Pos  {f3(ukf_pos)}']
            if vel is not None:
                ukf_speed_cur = float(np.linalg.norm(vel))
                lines += [f'Vel  |v|={ukf_speed_cur:.3f} m/s']
            else:
                lines += ['Vel  N/A']
            lines += [f'Shft {f3s(ukf_shift)}']
            if gt_pos is not None:
                ukf_err = np.linalg.norm(ukf_pos[:3] - gt_pos[:3])
                lines += [f'ErrP {ukf_err:.3f} m']
            lines += ['']

        info_text_obj.set_text('\n'.join(lines))
        fig.canvas.draw_idle()

    def animate(_):
        if animation_state['is_playing']:
            nv = slider.val + 0.05
            slider.set_val(nv if nv <= t_max else 0)
        return []

    def toggle_animation(event):
        animation_state['is_playing'] = not animation_state['is_playing']
        if animation_state['is_playing']:
            button.label.set_text('Pause')
            if animation_state['timer'] is None:
                animation_state['timer'] = fig.canvas.new_timer(interval=50)
                animation_state['timer'].add_callback(animate, None)
            animation_state['timer'].start()
        else:
            button.label.set_text('Play')
            if animation_state['timer'] is not None:
                animation_state['timer'].stop()

    def on_prev(_):
        idx = int(np.argmin(np.abs(timestamps_norm - slider.val)))
        idx = max(0, idx - 1)
        slider.set_val(float(timestamps_norm[idx]))

    def on_next(_):
        idx = int(np.argmin(np.abs(timestamps_norm - slider.val)))
        idx = min(len(timestamps_norm) - 1, idx + 1)
        slider.set_val(float(timestamps_norm[idx]))

    button.on_clicked(toggle_animation)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    slider.on_changed(update)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if is_3d:
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
    else:
        ax.set_title('2D Trajectory')

    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)
    if not is_3d:
        ax.axis('equal')

    update(0)
    plt.show()
    return fig, ax


def display_frame_window(frame, p0=None, p1=None, window_name="Current Frame",
                         trajectory_info=None, wait_key=1):
    """
    Display the current frame in an OpenCV window with optional feature tracking visualization.
    """
    if len(frame.shape) == 2:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # pylint: disable=no-member
    else:
        display_frame = frame.copy()

    if p0 is not None and p1 is not None:
        if len(p0.shape) == 3:
            p0 = p0.reshape(-1, 2)
        if len(p1.shape) == 3:
            p1 = p1.reshape(-1, 2)
        for pt0, pt1 in zip(p0, p1):
            pt0_tuple = tuple(pt0.astype(int))
            pt1_tuple = tuple(pt1.astype(int))
            cv2.circle(display_frame, pt0_tuple, 1, (0, 255, 255), -1)  # pylint: disable=no-member
            cv2.line(display_frame, pt0_tuple, pt1_tuple, (0, 255, 0), 1)  # pylint: disable=no-member
            cv2.circle(display_frame, pt1_tuple, 1, (0, 0, 255), -1)  # pylint: disable=no-member
    elif p1 is not None:
        if len(p1.shape) == 3:
            p1 = p1.reshape(-1, 2)
        for pt in p1:
            cv2.circle(display_frame, tuple(pt.astype(int)), 3, (255, 255, 0), -1)  # pylint: disable=no-member

    display_frame = cv2.resize(display_frame, (0, 0), fx=3, fy=3)  # pylint: disable=no-member
    cv2.imshow(window_name, display_frame)  # pylint: disable=no-member
    return cv2.waitKey(wait_key)  # pylint: disable=no-member


def close_all_windows():
    """Close all OpenCV windows."""
    cv2.destroyAllWindows()  # pylint: disable=no-member


import open3d as o3d


# =========================
# Sample trajectory (Nx3)
# =========================
_sample_points = np.array([
    [0.0, 0.0, 0.0], [0.5, 0.2, 0.1], [1.0, 0.4, 0.15],
    [1.5, 0.7, 0.2], [2.0, 1.0, 0.25], [2.5, 1.3, 0.3], [3.0, 1.6, 0.35],
])


def create_grid(size=5, step=1.0):
    pts, lines, idx = [], [], 0
    for i in np.arange(-size, size + step, step):
        pts += [[i, -size, 0], [i, size, 0]];  lines.append([idx, idx+1]); idx += 2
        pts += [[-size, i, 0], [size, i, 0]];  lines.append([idx, idx+1]); idx += 2
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(pts)
    grid.lines  = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]] * len(lines))
    return grid


def create_trajectory(points):
    lines = [[i, i+1] for i in range(len(points)-1)]
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(points)
    traj.lines  = o3d.utility.Vector2iVector(lines)
    traj.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))
    return traj


def create_drone_marker(position, radius=0.08):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(position)
    return sphere


class TrajectoryViewer:
    def __init__(self, points):
        self.points = points
        self.idx    = 0
        self.vis    = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Drone Trajectory Viewer")
        self.grid   = create_grid()
        self.traj   = create_trajectory(points)
        self.marker = create_drone_marker(self.points[self.idx])
        for g in (self.grid, self.traj, self.marker):
            self.vis.add_geometry(g)
        self.vis.register_key_callback(262, self.next_frame)
        self.vis.register_key_callback(263, self.prev_frame)
        print("Controls:  → next time step   ← previous time step")

    def update_marker(self):
        self.vis.remove_geometry(self.marker, reset_bounding_box=False)
        self.marker = create_drone_marker(self.points[self.idx])
        self.vis.add_geometry(self.marker, reset_bounding_box=False)

    def next_frame(self, vis):
        if self.idx < len(self.points) - 1:
            self.idx += 1; self.update_marker()
        return False

    def prev_frame(self, vis):
        if self.idx > 0:
            self.idx -= 1; self.update_marker()
        return False

    def run(self):
        self.vis.run(); self.vis.destroy_window()


if __name__ == "__main__":
    print("Starting Open3D viewer...")
    viewer = TrajectoryViewer(_sample_points)
    viewer.run()
