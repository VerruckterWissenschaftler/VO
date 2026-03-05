import cv2
import logging
import numpy as np
import yaml
import os
from src.ConfigManager import ConfigManager
from src.DataManager import DataManager
from src.FeaturesFlow import get_matcher
from src.Plotter import display_frame_window, close_all_windows
from src.HeightEstimator import HeightEstimator
from src.UKF import UKF
from tqdm import tqdm


class VOEstimator:
    def __init__(self, dataset_path, calib_path):
        """
        Initialize Visual Odometry Estimator.

        Parameters
        ----------
        dataset_path : str  Path to dataset directory (must contain params.yml, CSVs).
        calib_path   : str  Path to calibration directory (YAML files).
        """
        self.logger = logging.getLogger("VOEstimator")
        params = self._load_dataset_params(dataset_path)

        self.use_imu = params.get("use_imu", True)
        self.use_vo = params.get("use_vo", True)
        self.use_ukf = params.get("use_ukf", True)
        start_time = params.get("start_time", None)
        duration = params.get("duration", float("inf"))
        self.show_frames = params.get("show_frames", False)
        self.matcher_name = params.get("matcher", "sift")

        self.frame_count = 0
        self.imu_update_count = 0

        self.p0 = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.shift = None
        self.prev_frame = None
        self.curr_frame = None

        self.init_height = params.get("init_height", None)

        self.config_manager = ConfigManager(calib_path)
        self.data_manager = DataManager(
            dataset_path, duration=duration, use_imu=self.use_imu, start_time=start_time
        )
        pose, orientation = self.data_manager.get_first_init_groundtruth_pose()

        if self.init_height is not None:
            pose = pose.copy()
            self.logger.info(
                "Overriding initial height: GT Z=%.4f → init_height=%.4f",
                pose[2], self.init_height,
            )
            pose[2] = float(self.init_height)
        self.current_pose = pose
        self.current_orientation = orientation
        self.config = self.config_manager.config
        self.R = None
        self.shifts = {}
        self.orientations = {}

        self.last_imu_time = None
        self.imu_orientation = orientation.copy()
        self._prev_orientation = None   # used to compute relative rotation for E decomposition

        # Incremental position accumulator (used by UKF update step)
        self._cumulative_position = pose.copy()

        # UKF cumulative position (accumulated from filtered shifts externally)
        self._ukf_cumulative_position = pose.copy()
        self._last_frame_time = None

        # Height-based scale estimator
        self.height_estimator = HeightEstimator()

        # UKF — 6-state filter: [vx, vy, vz, roll, pitch, yaw]
        # Predict: IMU gyro (orientation) + linear accel (velocity)
        # Update:  VO per-frame shift (sparse, ~20 Hz)
        self.ukf = UKF(
            process_noise_vel=1e-4,    # per IMU step (~200 Hz) — keeps covariance sensible
            process_noise_angle=1e-5,
            measurement_noise=0.5,
            vel_decay_rate=2.0,        # velocity e-folds in 0.5 s — prevents accel-bias runaway
        )
        self.ukf.initialize(orientation)
        self.ukf_positions = {}     # timestamp → cumulative filtered position
        self.ukf_orientations = {}  # timestamp → [roll, pitch, yaw] radians
        self.gt_positions_frames = {}   # timestamp → GT position at that image frame
        self.gt_orientations_frames = {}  # timestamp → GT quaternion [x,y,z,w]

        # Pre-extract GT arrays for fast per-frame lookup
        gt_df = self.data_manager.gt_df
        self._gt_times = gt_df["Time"].values.astype(np.float64) if len(gt_df) > 0 else np.empty(0)
        self._gt_pos   = gt_df[["pose.position.x", "pose.position.y", "pose.position.z"]].values if len(gt_df) > 0 else np.empty((0, 3))
        self._gt_quat  = gt_df[["pose.orientation.x", "pose.orientation.y",
                                 "pose.orientation.z", "pose.orientation.w"]].values if len(gt_df) > 0 else np.empty((0, 4))

        # Matcher function (lazy-loaded GPU models via get_matcher)
        self._matcher = get_matcher(self.matcher_name)

        self.setup_camera()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _lookup_gt_at(self, t: float):
        """Return (position, quaternion) from GT closest to timestamp t."""
        if len(self._gt_times) == 0:
            return np.zeros(3), np.array([0., 0., 0., 1.])
        idx = int(np.argmin(np.abs(self._gt_times - t)))
        return self._gt_pos[idx].copy(), self._gt_quat[idx].copy()

    def _load_dataset_params(self, dataset_path):
        params_file = os.path.join(dataset_path, "params.yml")
        if os.path.exists(params_file):
            try:
                with open(params_file, "r", encoding="utf-8") as f:
                    params = yaml.safe_load(f)
                    self.logger.info("Loaded params from %s: %s", params_file, params)
                    return params if params is not None else {}
            except (OSError, yaml.YAMLError) as e:
                self.logger.warning("Failed to load params from %s: %s", params_file, e)
        return {}

    def setup_camera(self):
        self.camera_matrix = self.config_manager.get_camera_matrix()
        self.dist_coeffs = self.config_manager.get_distortion_coeffs()

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def undistort_image(self, image):
        K = self.camera_matrix
        D = self.dist_coeffs
        DIM = (image.shape[1], image.shape[0])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(  # pylint: disable=no-member
            K, D, np.eye(3), K, DIM, cv2.CV_16SC2
        )
        return cv2.remap(image, map1, map2,  # pylint: disable=no-member
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def derive_essential_matrix(self, p0_good, p1_good):
        x0 = p0_good.reshape(-1, 2)
        x1 = p1_good.reshape(-1, 2)
        Kinv = np.linalg.inv(self.camera_matrix)
        x0 = (Kinv @ np.hstack([x0, np.ones((x0.shape[0], 1))]).T)
        x1 = (Kinv @ np.hstack([x1, np.ones((x1.shape[0], 1))]).T)
        x0 = (x0[:2] / x0[2]).T
        x1 = (x1[:2] / x1[2]).T
        E, mask = cv2.findEssentialMat(  # pylint: disable=no-member
            x1, x0, focal=1.0, pp=(0, 0),
            method=cv2.RANSAC, prob=0.999, threshold=0.01,
        )
        return E, mask, x0, x1

    def estimate_pose(self, p0_good, p1_good):
        E, mask, x0, x1 = self.derive_essential_matrix(p0_good, p1_good)

        if E is None:
            return np.zeros(3)

        inlier_count = int(np.sum(mask)) if mask is not None else len(x0)
        if inlier_count < 6:
            self.logger.debug("Low inlier count (%d) for pose estimation.", inlier_count)
            return np.zeros(3)

        # Current absolute rotation: prefer IMU, fall back to UKF, then stored
        if self.use_imu and self.imu_update_count > 0:
            R_curr = self.imu_orientation
        elif self.use_ukf and self.ukf.initialized:
            R_curr = self.ukf.get_rotation_matrix()
        else:
            R_curr = self.current_orientation

        # recoverPose selects the correct t sign via cheirality (its implementation is proven).
        # We keep its t but discard its R — the IMU/UKF provides a better absolute rotation.
        try:
            _, _, t, _ = cv2.recoverPose(E, x1, x0, focal=1.0, pp=(0, 0))  # pylint: disable=no-member
        except cv2.error:  # pylint: disable=no-member
            return np.zeros(3)
        t_unit = np.array(t).flatten()

        self._prev_orientation = R_curr.copy()

        # Rotate unit translation into world frame using IMU/UKF orientation
        t_world = R_curr @ t_unit
        self.current_orientation = R_curr.copy()

        scale = self.height_estimator.compute_scale(t_world[2])
        return t_world * scale

    @staticmethod
    def _is_static(
        p0_good: np.ndarray,
        p1_good: np.ndarray,
        pixel_threshold: float = 1.5,
        static_ratio: float = 0.80,
    ) -> bool:
        """Return True if ≥80 % of feature tracks have sub-pixel displacement."""
        displacements = np.linalg.norm(
            p1_good.reshape(-1, 2) - p0_good.reshape(-1, 2), axis=1
        )
        return float(np.mean(displacements < pixel_threshold)) > static_ratio

    @staticmethod
    def _cheirality_select_t(
        R: np.ndarray,
        t_unit: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
    ) -> np.ndarray:
        """
        Return t_unit or -t_unit so that triangulated points are in front
        of both cameras (cheirality / positive-depth test).
        """
        if len(x0) == 0:
            return t_unit

        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])

        def _count_front(t_vec: np.ndarray) -> int:
            P1 = np.hstack([R, t_vec.reshape(3, 1)])
            count = 0
            for p0, p1 in zip(x0, x1):
                A = np.array([
                    p0[0] * P0[2] - P0[0],
                    p0[1] * P0[2] - P0[1],
                    p1[0] * P1[2] - P1[0],
                    p1[1] * P1[2] - P1[1],
                ])
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                if abs(X[3]) < 1e-10:
                    continue
                X3d = X[:3] / X[3]
                X3d_c1 = R @ X3d + t_vec
                if X3d[2] > 0 and X3d_c1[2] > 0:
                    count += 1
            return count

        return t_unit if _count_front(t_unit) >= _count_front(-t_unit) else -t_unit

    # ------------------------------------------------------------------
    # IMU integration
    # ------------------------------------------------------------------

    def update_attitude(self, imu_event):
        """
        Integrate IMU angular velocity (orientation) and linear acceleration (UKF + height).
        """
        imu_data = imu_event["data"]
        current_time = imu_event["time"]

        if self.last_imu_time is None:
            self.last_imu_time = current_time
            return

        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        if dt <= 0 or dt > 0.1:
            return

        # --- Angular velocity → orientation update (Rodrigues) ---
        omega = np.array([
            imu_data.get("angular_velocity.x", 0.0),
            imu_data.get("angular_velocity.y", 0.0),
            imu_data.get("angular_velocity.z", 0.0),
        ])
        angle = np.linalg.norm(omega) * dt
        if angle >= 1e-8:
            axis = omega / np.linalg.norm(omega)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            self.imu_orientation = self.imu_orientation @ R_delta
            self.imu_update_count += 1

            # Periodic re-orthonormalization
            if np.random.random() < 0.01:
                U, _, Vt = np.linalg.svd(self.imu_orientation)
                self.imu_orientation = U @ Vt

        if self.imu_update_count % 100 == 0 and self.imu_update_count > 0:
            self.logger.debug("IMU updates: %d, dt: %.4fs", self.imu_update_count, dt)

        # --- Linear acceleration → height estimator + UKF prediction ---
        a_body = np.array([
            imu_data.get("linear_acceleration.x", 0.0),
            imu_data.get("linear_acceleration.y", 0.0),
            imu_data.get("linear_acceleration.z", 0.0),
        ])

        # Use current world rotation (prefer IMU orientation if available)
        R_world = self.imu_orientation if self.imu_update_count > 0 else self.current_orientation

        # Update height estimator
        self.height_estimator.update_imu(a_body, R_world, dt)

        # UKF prediction step: gyro (orientation) + linear acceleration (velocity)
        if self.use_ukf:
            self.ukf.predict(omega, a_body, dt)
            # IMU-only mode: accumulate UKF position at IMU rate (more accurate than per-frame)
            if not self.use_vo and self.ukf.initialized:
                self._ukf_cumulative_position += self.ukf.get_filtered_shift(dt)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, frame, event_time=None):
        frame = self.undistort_image(frame)
        if self.prev_frame is None:
            self.prev_frame = frame
            self._last_frame_time = event_time
            return

        # Inter-frame time delta
        dt = 0.0
        if event_time is not None and self._last_frame_time is not None:
            dt = event_time - self._last_frame_time
        self._last_frame_time = event_time

        self.curr_frame = frame
        tracked = self._matcher(self.prev_frame, self.curr_frame)
        self.prev_frame = self.curr_frame

        if tracked is None or len(tracked[1]) < 10:
            self.p0 = None
            return

        p0_good, p1_good = tracked
        self.p0 = p1_good.reshape(-1, 1, 2)

        if self._is_static(p0_good, p1_good):
            # Camera is stationary — skip VO and apply zero-velocity constraint
            self.shift = np.zeros(3)
            if self.use_ukf:
                self.ukf.zupt_update()
        else:
            self.shift = self.estimate_pose(p0_good, p1_good)
            self.shift /= 1e1
            self._cumulative_position = self._cumulative_position + self.shift
            if self.use_ukf and dt > 0:
                self.ukf.update(self.shift, dt)

        # Accumulate UKF trajectory externally from filtered shift
        if self.use_ukf and dt > 0 and self.ukf.initialized:
            self._ukf_cumulative_position += self.ukf.get_filtered_shift(dt)

        if self.show_frames:
            current_pos = self._cumulative_position
            info = {
                "Frame": self.frame_count,
                "Features": len(p1_good),
                "Position": f"[{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]",
                "Height": f"{self.height_estimator.get_height():.3f}m",
                "Scale": f"{self.height_estimator.scale:.4f}",
                "Matcher": self.matcher_name,
            }
            if event_time is not None:
                info["Time"] = f"{event_time:.2f}s"
            key = display_frame_window(
                self.curr_frame, p0_good, p1_good,
                window_name="Visual Odometry - Current Frame",
                trajectory_info=info,
                wait_key=1,
            )
            if key == 27:
                self.show_frames = False
                close_all_windows()

        self.frame_count += 1

    # ------------------------------------------------------------------
    # Trajectory calculation
    # ------------------------------------------------------------------

    def calculate_trajectory(self):
        """
        Build final trajectories from stored per-frame shifts and UKF positions.

        Returns
        -------
        (vo_timestamps, vo_positions, vo_orientations,
         ukf_timestamps, ukf_positions)
        """
        sorted_times = sorted(self.shifts.keys())

        # Raw VO trajectory: cumulative sum of (already-scaled) shifts + initial pose
        estimated_positions = [self.shifts[t] for t in sorted_times]
        vo_trajectory = np.cumsum(estimated_positions, axis=0) + self.current_pose
        vo_orientations = np.array([self.orientations.get(t, np.eye(3)) for t in sorted_times])

        # UKF trajectory (stored incrementally in self.ukf_positions)
        ukf_times = sorted(self.ukf_positions.keys())
        ukf_positions = np.array([self.ukf_positions[t] for t in ukf_times]) if ukf_times else np.empty((0, 3))
        ukf_euler = np.array([self.ukf_orientations.get(t, np.zeros(3)) for t in ukf_times]) if ukf_times else np.empty((0, 3))

        # GT sampled at image-frame timestamps (aligned with VO/UKF)
        gt_frame_times = sorted(self.gt_positions_frames.keys())
        gt_frame_positions = np.array([self.gt_positions_frames[t] for t in gt_frame_times]) if gt_frame_times else np.empty((0, 3))
        gt_frame_orientations = np.array([self.gt_orientations_frames[t] for t in gt_frame_times]) if gt_frame_times else np.empty((0, 4))

        return (
            sorted_times, vo_trajectory, vo_orientations,
            ukf_times, ukf_positions, ukf_euler,
            gt_frame_times, gt_frame_positions, gt_frame_orientations,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        """
        Process all events. Returns VO and UKF trajectories.

        Returns
        -------
        vo_timestamps     : list of float
        vo_positions      : (N, 3) ndarray
        vo_orientations   : (N, 3, 3) ndarray
        ukf_timestamps    : list of float
        ukf_positions     : (M, 3) ndarray
        """
        try:
            for event in tqdm(self.data_manager, desc="Processing events", unit="event"):
                event_type = event["type"]
                event_time = event["time"]

                if event_type == "image":
                    if self.use_vo:
                        self.update(event["data"]["image"], event_time=event_time)
                    if self.shift is None:
                        self.shift = np.zeros(3)
                    self.shifts[event_time] = self.shift if self.use_vo else np.zeros(3)
                    self.orientations[event_time] = self.current_orientation.copy()
                    # Snapshot UKF state at every image frame
                    if self.use_ukf:
                        self.ukf_positions[event_time] = self._ukf_cumulative_position.copy()
                        self.ukf_orientations[event_time] = self.ukf.x[3:].copy()  # [roll, pitch, yaw]
                    # Snapshot nearest GT at this image frame (aligned timestamps)
                    gt_p, gt_q = self._lookup_gt_at(event_time)
                    self.gt_positions_frames[event_time] = gt_p
                    self.gt_orientations_frames[event_time] = gt_q

                elif event_type == "imu":
                    self.update_attitude(event)

        finally:
            if self.show_frames:
                close_all_windows()

        if self.imu_update_count > 0:
            self.logger.info("IMU integration: %d updates", self.imu_update_count)
        else:
            self.logger.warning("No IMU data was processed")

        self.logger.info(
            "Height estimator: final height=%.3fm, scale=%.4f, IMU updates=%d",
            self.height_estimator.get_height(),
            self.height_estimator.scale,
            self.height_estimator.imu_updates,
        )

        return self.calculate_trajectory()
