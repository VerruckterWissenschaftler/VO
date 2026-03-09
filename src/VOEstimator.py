import cv2
import logging
import numpy as np
import yaml
import os
from src.ConfigManager import ConfigManager
from src.DataManager import DataManager
from src.FeaturesFlow import get_matcher
from src.ukf import UKF
from src.visualization import UKFDebugger, display_frame_window, close_all_windows
from tqdm import tqdm


class VOEstimator:
    def __init__(self, dataset_path, calib_path):
        """
        Initialize Visual Odometry Estimator.

        UKF is always active. VO measurements are fused into it when use_vo=True.
        IMU integration (Rodrigues orientation) lives inside UKF.feed_imu().
        """
        self.logger = logging.getLogger("VOEstimator")
        params = self._load_dataset_params(dataset_path)

        self.use_imu      = params.get("use_imu", True)
        self.use_vo       = params.get("use_vo", True)
        self.debug_ukf    = params.get("debug_ukf", False)
        self.use_gt_imu   = params.get("use_gt_imu", False)
        start_time        = params.get("start_time", None)
        duration          = params.get("duration", float("inf"))
        self.show_frames  = params.get("show_frames", False)
        self.matcher_name = params.get("matcher", "sift")
        self.init_height  = params.get("init_height", None)

        self.frame_count = 0
        self.p0          = None
        self.prev_frame  = None
        self.curr_frame  = None
        self.last_imu_time    = None
        self._last_frame_time = None

        self.config_manager = ConfigManager(calib_path)
        self.data_manager   = DataManager(
            dataset_path, duration=duration, use_imu=self.use_imu, start_time=start_time
        )
        pose, orientation = self.data_manager.get_first_init_groundtruth_pose()

        if self.init_height is not None:
            pose = pose.copy()
            self.logger.info("Overriding initial height: GT Z=%.4f → init_height=%.4f",
                             pose[2], self.init_height)
            pose[2] = float(self.init_height)

        self.config = self.config_manager.config

        # VO cumulative position (feeds UKF as sparse position measurement)
        self._cumulative_position = pose.copy()

        # UKF — 12-state: [px,py,pz, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
        # IMU integration and orientation tracking live inside UKF.feed_imu().
        # VO position is fed via ukf.vo_update() at image-frame rate.
        imu_params = self.config_manager.get_imu_params()
        self.ukf = UKF(
            accel_noise_density=imu_params['accelerometer_noise_density'],
            accel_random_walk=imu_params['accelerometer_random_walk'],
            gyro_noise_density=imu_params['gyroscope_noise_density'],
            gyro_random_walk=imu_params['gyroscope_random_walk'],
            meas_noise_vo=0.5,
            meas_noise_gyro=1e-3,
            vel_decay_rate=2.0,
            log_path=os.path.join(dataset_path, "ukf_accel_log.csv"),
        )
        self.ukf.initialize(orientation, initial_position=pose)
        self.ukf_debugger = UKFDebugger(self.ukf) if self.debug_ukf else None

        self.ukf_positions    = {}  # timestamp → UKF position (3,)
        self.ukf_velocities   = {}  # timestamp → UKF velocity [vx,vy,vz] m/s
        self.ukf_orientations = {}  # timestamp → [roll, pitch, yaw] radians

        self.gt_positions_frames    = {}  # timestamp → GT position (3,)
        self.gt_orientations_frames = {}  # timestamp → GT quaternion [x,y,z,w]

        # Camera frames and IMU accel log (for plotter side-panels)
        self.frames:   dict = {}   # timestamp → undistorted frame
        self._imu_ts:  list = []   # IMU absolute timestamps
        self._imu_acc: list = []   # IMU body-frame accelerations (3,)

        # Pre-extract GT arrays for fast per-frame lookup
        gt_df = self.data_manager.gt_df
        self._gt_times = gt_df["Time"].values.astype(np.float64) if len(gt_df) > 0 else np.empty(0)
        self._gt_pos   = gt_df[["pose.position.x", "pose.position.y", "pose.position.z"]].values \
                         if len(gt_df) > 0 else np.empty((0, 3))
        self._gt_quat  = gt_df[["pose.orientation.x", "pose.orientation.y",
                                 "pose.orientation.z", "pose.orientation.w"]].values \
                         if len(gt_df) > 0 else np.empty((0, 4))

        # Precompute GT velocity and acceleration (double finite-diff) for use_gt_imu
        self._gt_acc_times = np.empty(0)
        self._gt_acc       = np.empty((0, 3))
        if self.use_gt_imu and len(self._gt_times) > 2:
            gt_dt      = np.maximum(np.diff(self._gt_times), 1e-9)
            gt_vel     = np.diff(self._gt_pos, axis=0) / gt_dt[:, None]
            gt_vel_t   = self._gt_times[1:]
            gt_vel_dt  = np.maximum(np.diff(gt_vel_t), 1e-9)
            self._gt_acc       = np.diff(gt_vel, axis=0) / gt_vel_dt[:, None]
            self._gt_acc_times = gt_vel_t[1:]

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
        self.dist_coeffs   = self.config_manager.get_distortion_coeffs()

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def undistort_image(self, image):
        K   = self.camera_matrix
        D   = self.dist_coeffs
        DIM = (image.shape[1], image.shape[0])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(  # pylint: disable=no-member
            K, D, np.eye(3), K, DIM, cv2.CV_16SC2
        )
        return cv2.remap(image, map1, map2,  # pylint: disable=no-member
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def derive_essential_matrix(self, p0_good, p1_good):
        x0   = p0_good.reshape(-1, 2)
        x1   = p1_good.reshape(-1, 2)
        Kinv = np.linalg.inv(self.camera_matrix)
        x0   = (Kinv @ np.hstack([x0, np.ones((x0.shape[0], 1))]).T)
        x1   = (Kinv @ np.hstack([x1, np.ones((x1.shape[0], 1))]).T)
        x0   = (x0[:2] / x0[2]).T
        x1   = (x1[:2] / x1[2]).T
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

        # Use UKF's Rodrigues-integrated orientation for both t-sign and world-frame rotation
        R_curr = self.ukf.imu_orientation
        try:
            _, _, t, _ = cv2.recoverPose(E, x1, x0, focal=1.0, pp=(0, 0))  # pylint: disable=no-member
        except cv2.error:  # pylint: disable=no-member
            return np.zeros(3)

        return R_curr @ np.array(t).flatten()

    # ------------------------------------------------------------------
    # IMU integration  (all state manipulation inside ukf.feed_imu)
    # ------------------------------------------------------------------

    def update_attitude(self, imu_event):
        """Process one IMU event: forward omega + accel to UKF."""
        imu_data     = imu_event["data"]
        current_time = imu_event["time"]

        if self.last_imu_time is None:
            self.last_imu_time = current_time
            return

        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        if dt <= 0 or dt > 0.1:
            return

        omega = np.array([
            imu_data.get("angular_velocity.x", 0.0),
            imu_data.get("angular_velocity.y", 0.0),
            imu_data.get("angular_velocity.z", 0.0),
        ])
        a_body = np.array([
            imu_data.get("linear_acceleration.x", 0.0),
            imu_data.get("linear_acceleration.y", 0.0),
            imu_data.get("linear_acceleration.z", 0.0),
        ])

        self._imu_ts.append(current_time)
        self._imu_acc.append(a_body.copy())

        gt_R          = None
        gt_accel_world = None
        if self.use_gt_imu:
            _, gt_quat = self._lookup_gt_at(current_time)
            x, y, z, w = gt_quat
            gt_R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ])
            if len(self._gt_acc_times) > 0:
                idx = int(np.argmin(np.abs(self._gt_acc_times - current_time)))
                gt_accel_world = self._gt_acc[idx].copy()

        self.ukf.feed_imu(omega, a_body, dt, gt_orientation=gt_R, gt_accel_world=gt_accel_world)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, frame, event_time=None):
        frame = self.undistort_image(frame)
        if event_time is not None:
            self.frames[event_time] = frame.copy()

        if self.prev_frame is None:
            self.prev_frame = frame
            self._last_frame_time = event_time
            return

        self._last_frame_time = event_time
        self.curr_frame = frame
        tracked = self._matcher(self.prev_frame, self.curr_frame)
        self.prev_frame = self.curr_frame

        if not self.use_vo:
            return

        if tracked is None or len(tracked[1]) < 10:
            self.p0 = None
            return

        p0_good, p1_good = tracked
        self.p0 = p1_good.reshape(-1, 1, 2)

        shift = self.estimate_pose(p0_good, p1_good)
        shift /= 1e1
        self._cumulative_position = self._cumulative_position + shift
        self.ukf.vo_update(self._cumulative_position)

        if self.show_frames:
            pos  = self.ukf.get_position()
            info = {
                "Frame":    self.frame_count,
                "Features": len(p1_good),
                "Position": f"[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]",
                "Matcher":  self.matcher_name,
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
    # Trajectory
    # ------------------------------------------------------------------

    def calculate_trajectory(self):
        """
        Build final trajectories from per-frame UKF snapshots.

        Returns
        -------
        ukf_timestamps          : list of float
        ukf_positions           : (N, 3) ndarray
        ukf_euler               : (N, 3) ndarray  [roll, pitch, yaw] radians
        gt_frame_timestamps     : list of float
        gt_frame_positions      : (N, 3) ndarray
        gt_frame_orientations   : (N, 4) ndarray  quaternion [x,y,z,w]
        """
        ukf_times = sorted(self.ukf_positions.keys())
        ukf_positions   = np.array([self.ukf_positions[t]              for t in ukf_times]) \
                          if ukf_times else np.empty((0, 3))
        ukf_velocities  = np.array([self.ukf_velocities.get(t, np.zeros(3)) for t in ukf_times]) \
                          if ukf_times else np.empty((0, 3))
        ukf_euler       = np.array([self.ukf_orientations.get(t, np.zeros(3)) for t in ukf_times]) \
                          if ukf_times else np.empty((0, 3))

        gt_frame_times = sorted(self.gt_positions_frames.keys())
        gt_frame_positions = np.array([self.gt_positions_frames[t] for t in gt_frame_times]) \
                             if gt_frame_times else np.empty((0, 3))
        gt_frame_orientations = np.array([self.gt_orientations_frames[t] for t in gt_frame_times]) \
                                if gt_frame_times else np.empty((0, 4))

        return (
            ukf_times, ukf_positions, ukf_velocities, ukf_euler,
            gt_frame_times, gt_frame_positions, gt_frame_orientations,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        """
        Process all events. UKF predicts from IMU at full rate; VO measurements
        are fused in when available (use_vo=True). Returns UKF trajectory vs GT.
        """
        try:
            for event in tqdm(self.data_manager, desc="Processing events", unit="event"):
                event_type = event["type"]
                event_time = event["time"]

                if event_type == "image":
                    self.update(event["data"]["image"], event_time=event_time)
                    # Snapshot UKF state at every image frame
                    self.ukf_positions[event_time]    = self.ukf.get_position()
                    self.ukf_velocities[event_time]   = self.ukf.get_velocity()
                    self.ukf_orientations[event_time] = self.ukf.x[6:9].copy()
                    # Nearest GT at this frame
                    gt_p, gt_q = self._lookup_gt_at(event_time)
                    self.gt_positions_frames[event_time]    = gt_p
                    self.gt_orientations_frames[event_time] = gt_q

                elif event_type == "imu":
                    if self.use_imu:
                        self.update_attitude(event)

        finally:
            if self.show_frames:
                close_all_windows()
            self.ukf.close_log()

        if self.ukf._imu_update_count > 0:
            self.logger.info("IMU integration: %d updates", self.ukf._imu_update_count)
        else:
            self.logger.warning("No IMU data was processed")

        if self.ukf_debugger is not None:
            self.ukf_debugger.summary()
            self.ukf_debugger.plot(
                gt_timestamps=self._gt_times,
                gt_positions=self._gt_pos,
                gt_quaternions=self._gt_quat,
            )

        return self.calculate_trajectory()
