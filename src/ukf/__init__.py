import csv
from collections import deque
import numpy as np
from scipy import signal
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rotation_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Extract ZYX Euler angles [roll, pitch, yaw] from rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw])


def _euler_zyx_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build 3×3 rotation matrix from ZYX Euler angles."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr               ],
    ])


# Indices of angle states — need circular mean / wrapping
_ANGLE_IDX = [6, 7, 8]

_G = 9.81  # gravitational acceleration, m/s²


# ---------------------------------------------------------------------------
# filterpy callbacks — state space
# ---------------------------------------------------------------------------

def _state_mean_fn(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    """Weighted mean with circular mean for angle states."""
    x_mean = sigmas.T @ Wm
    for idx in _ANGLE_IDX:
        x_mean[idx] = np.arctan2(
            np.sum(Wm * np.sin(sigmas[:, idx])),
            np.sum(Wm * np.cos(sigmas[:, idx])),
        )
    return x_mean


def _residual_x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """State residual with angle wrapping for indices 6,7,8."""
    r = a - b
    for idx in _ANGLE_IDX:
        r[idx] = _wrap_angle(r[idx])
    return r


# ---------------------------------------------------------------------------
# filterpy callbacks — orientation measurement space (all 3 components are angles)
# ---------------------------------------------------------------------------

def _ang_mean_fn(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    """Circular mean for a 3-component angle measurement."""
    return np.array([
        np.arctan2(np.sum(Wm * np.sin(sigmas[:, i])),
                   np.sum(Wm * np.cos(sigmas[:, i])))
        for i in range(3)
    ])


def _residual_ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual with wrapping for all 3 angle components."""
    return np.array([_wrap_angle(a[i] - b[i]) for i in range(3)])


class UKF:
    """
    9-state UKF for fused IMU + VO navigation, built on filterpy.

    State vector: x = [px, py, pz,  vx, vy, vz,  roll, pitch, yaw]
                       0   1   2     3   4   5     6     7      8

    Angular velocity (omega) and linear acceleration are external inputs, not state.

    Data ingestion:
      append_accelerometer(a_body) — store latest IMU linear acceleration.
      append_orientation(R)        — measurement update: IMU orientation → angle states.

    Predict / update:
      predict(omega, dt) — propagate state using stored accelerometer + provided omega.
      vo_update(pos)     — sparse VO position measurement for pos states [0:3].
    """

    def __init__(
        self,
        accel_noise_density:    float = 0.1,    # m/s²/√Hz  — from IMU calibration
        accel_random_walk:      float = 0.002,  # m/s²√Hz   — accel bias instability
        gyro_noise_density:     float = 0.05,   # rad/s/√Hz — from IMU calibration
        gyro_random_walk:       float = 4e-5,   # rad/s√Hz  — gyro bias instability
        meas_noise_vo:          float = 0.5,
        meas_noise_orientation: float = 1e-2,
        initial_uncertainty:    float = 1e-3,
        vel_decay_rate:         float = 2.0,
        accel_kernel_size:      int   = 1,
        log_path:               str | None = None,
    ):
        self.vel_decay_rate          = vel_decay_rate
        self._meas_noise_vo          = meas_noise_vo
        self._meas_noise_orientation = meas_noise_orientation
        self.initialized = False

        # IMU noise power spectral densities (continuous-time)
        self._Na  = accel_noise_density ** 2   # m²/s³  — accel white noise PSD
        self._Nba = accel_random_walk   ** 2   # m²/s⁵  — accel bias random walk PSD
        self._Ng  = gyro_noise_density  ** 2   # rad²/s — gyro white noise PSD
        self._Nbg = gyro_random_walk    ** 2   # rad²/s³ — gyro bias random walk PSD

        n = 9
        self._pos = slice(0, 3)
        self._vel = slice(3, 6)
        self._ang = slice(6, 9)

        # External inputs — set before each predict call
        self._current_omega = np.zeros(3)

        points = MerweScaledSigmaPoints(
            n=n, alpha=1e-3, beta=2.0, kappa=0.0,
            subtract=_residual_x,
        )

        self._ukf = UnscentedKalmanFilter(
            dim_x=n, dim_z=3, dt=1.0,
            hx=self._hx_pos,   # overridden per update call
            fx=self._fx,
            points=points,
            x_mean_fn=_state_mean_fn,
            residual_x=_residual_x,
        )

        self._ukf.x = np.zeros(n)
        self._ukf.P = np.eye(n) * initial_uncertainty
        # Q is built dynamically in predict() from dt — initialise to placeholder
        self._ukf.Q = np.eye(n) * 1e-6
        self._ukf.R = np.eye(3) * meas_noise_vo

        # Rolling window of body-frame accelerometer readings; medfilt output is fed to _fx
        _k = accel_kernel_size if accel_kernel_size % 2 == 1 else accel_kernel_size + 1
        self._accel_kernel_size: int   = max(1, _k)
        self._accel_window:      deque = deque(maxlen=self._accel_kernel_size)

        # IMU-integrated orientation (Rodrigues), lives here rather than in VOEstimator
        self._imu_orientation = np.eye(3)
        self._imu_update_count = 0

        # Optional GT world-frame acceleration override (bypasses IMU accel + bias in _fx)
        self._gt_accel_world: np.ndarray | None = None

        # Acceleration log — writes world-frame accel (gravity-removed) after each predict step
        self._log_cumtime = 0.0
        self._log_file    = None
        self._log_writer  = None
        if log_path is not None:
            self._log_file   = open(log_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(["t_s", "ax_world", "ay_world", "az_world"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, R: np.ndarray | None = None,
                   initial_position: np.ndarray | None = None) -> None:
        """Set initial position and orientation from GT rotation matrix."""
        self._ukf.x[:] = 0.0
        if initial_position is not None:
            self._ukf.x[self._pos] = initial_position
        if R is not None:
            self._ukf.x[self._ang] = _rotation_to_euler_zyx(R)
            self._imu_orientation = R.copy()
        else:
            self._imu_orientation = np.eye(3)
        self._imu_update_count = 0
        self.initialized = True

    def append_accelerometer(self, accel_body: np.ndarray) -> None:
        """Push latest body-frame linear acceleration into the rolling window."""
        self._accel_window.append(accel_body)

    def append_orientation(self, R: np.ndarray) -> None:
        """
        Measurement update: fuse IMU-integrated orientation into angle states [6:9].

        Uses circular-mean and angle-wrapped residual so the update is correct
        across the ±π discontinuity.
        """
        if not self.initialized:
            return
        euler = _rotation_to_euler_zyx(R)

        # Temporarily override z_mean and residual_z for angle-aware update
        old_z_mean   = self._ukf.z_mean
        old_residual = self._ukf.residual_z
        self._ukf.z_mean   = _ang_mean_fn
        self._ukf.residual_z = _residual_ang

        self._ukf.update(
            euler,
            hx=self._hx_ang,
            R=np.eye(3) * self._meas_noise_orientation,
        )

        self._ukf.z_mean   = old_z_mean
        self._ukf.residual_z = old_residual

    def feed_imu(self, omega: np.ndarray, accel_body: np.ndarray, dt: float,
                 gt_orientation: np.ndarray | None = None,
                 gt_accel_world: np.ndarray | None = None,
                 update_filter: bool = True) -> None:
        """
        Full IMU step — call once per IMU event.

        1. Rodrigues-integrates omega → updates internal _imu_orientation (always).
        2. If update_filter=True: predict(omega, dt) + append_orientation().
           If update_filter=False: orientation-only mode — UKF state is not touched.

        Parameters
        ----------
        omega          : angular velocity [wx, wy, wz] rad/s (body frame)
        accel_body     : linear acceleration [ax, ay, az] m/s² (body frame, includes gravity)
        dt             : time step in seconds
        gt_orientation : optional GT rotation matrix; overrides IMU integration for the
                         orientation measurement update (use_gt_imu mode)
        gt_accel_world : optional GT world-frame acceleration (m/s²); when provided it is
                         used directly in _fx, bypassing IMU accel rotation and bias removal
        update_filter  : if False, only Rodrigues orientation is updated; UKF predict/update
                         steps are skipped (use when use_ukf=False)
        """
        if not self.initialized or dt <= 0 or dt > 0.5:
            return

        # Rodrigues orientation integration (always runs)
        angle = np.linalg.norm(omega) * dt
        if angle >= 1e-8:
            axis = omega / np.linalg.norm(omega)
            K = np.array([
                [0,       -axis[2],  axis[1]],
                [axis[2],  0,       -axis[0]],
                [-axis[1], axis[0],  0      ],
            ])
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            self._imu_orientation = self._imu_orientation @ R_delta
            self._imu_update_count += 1
            # Periodic re-orthonormalization
            if self._imu_update_count % 100 == 0:
                U, _, Vt = np.linalg.svd(self._imu_orientation)
                self._imu_orientation = U @ Vt

        if not update_filter:
            return

        # When GT accel is available, store it (already world-frame) in the window;
        # _fx will skip rotation. Otherwise store raw body-frame IMU accel.
        self._gt_accel_world = gt_accel_world
        self.predict(omega, dt)
        self._log_cumtime += dt
        if self._log_writer is not None:
            a = self._world_accel()
            self._log_writer.writerow([f"{self._log_cumtime:.6f}",
                                       f"{a[0]:.6f}", f"{a[1]:.6f}", f"{a[2]:.6f}"])
        self.append_orientation(gt_orientation if gt_orientation is not None
                                else self._imu_orientation)

    @property
    def imu_orientation(self) -> np.ndarray:
        """Rodrigues-integrated IMU orientation (3×3 rotation matrix)."""
        return self._imu_orientation

    def predict(self, omega: np.ndarray, dt: float) -> None:
        """
        Propagate state using stored accelerometer and provided angular velocity.
        Call append_accelerometer() before this each IMU step.

        omega  : angular velocity [wx, wy, wz] rad/s (body frame) — used directly in _fx
                 to integrate Euler angles; not stored in state.

        Q is rebuilt each call via Van Loan discretization so noise covariance
        scales correctly with the actual time step:

          Position–velocity block (per axis, from accel white noise Na):
            Q_pp = Na·dt³/3,  Q_pv = Q_vp = Na·dt²/2,  Q_vv = Na·dt

          Angle block (from gyro white noise Ng):
            Q_aa = Ng·dt
        """
        if not self.initialized or dt <= 0 or dt > 0.5:
            return

        self._current_omega = omega.copy()

        dt2 = dt * dt
        dt3 = dt2 * dt

        Q = np.zeros((9, 9))
        for i in range(3):
            # position states [0,1,2]
            Q[i,   i  ] = self._Na * dt3 / 3.0   # pos–pos
            Q[i,   i+3] = self._Na * dt2 / 2.0   # pos–vel
            Q[i+3, i  ] = self._Na * dt2 / 2.0   # vel–pos
            # velocity states [3,4,5]
            Q[i+3, i+3] = self._Na * dt           # vel–vel
            # angle states [6,7,8]
            Q[i+6, i+6] = self._Ng  * dt          # ang–ang (gyro white noise)

        self._ukf.Q = Q
        self._ukf.predict(dt=dt)

    def vo_update(self, position: np.ndarray) -> None:
        """Sparse measurement update: VO position → position states [0:3]."""
        if not self.initialized:
            return
        self._ukf.update(position, hx=self._hx_pos, R=np.eye(3) * self._meas_noise_vo)

    def close_log(self) -> None:
        """Flush and close the acceleration log file (if open)."""
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file   = None
            self._log_writer = None

    def _accel_filtered(self) -> np.ndarray:
        """
        Apply scipy.signal.medfilt along the time axis of the rolling window.

        kernel_size=[k, 1] filters each axis independently in one call.
        Returns the last (most recent) filtered sample.
        """
        if not self._accel_window:
            return np.zeros(3)
        arr = np.array(self._accel_window)          # (N, 3)
        k   = min(self._accel_kernel_size, len(arr))
        if k % 2 == 0:
            k -= 1
        k = max(k, 1)
        value = signal.medfilt(arr, kernel_size=[k, 1])[-1]
        return value

    def _world_accel(self) -> np.ndarray:
        """World-frame linear acceleration (gravity removed, medfilt) at current mean state."""
        if self._gt_accel_world is not None:
            return self._accel_filtered()
        roll, pitch, yaw = self._ukf.x[self._ang]
        # R = _euler_zyx_to_rotation(roll, pitch, yaw)
        # a = R @ self._accel_filtered()
        a = self._accel_filtered()
        # a[0] -= -0.2372
        # a[1] -= -1.5710
        # a[2] -= 9.7817
        return a

    def update_acc(self, accel_body: np.ndarray) -> None:
        """Alias for append_accelerometer — store latest body-frame linear acceleration."""
        self.append_accelerometer(accel_body)

    def update_orientation(self, R: np.ndarray) -> None:
        """Alias for append_orientation — fuse rotation matrix into angle states [6:9]."""
        self.append_orientation(R)

    def get_position(self) -> np.ndarray:
        return self._ukf.x[self._pos].copy()

    def get_velocity(self) -> np.ndarray:
        return self._ukf.x[self._vel].copy()

    def get_rotation_matrix(self) -> np.ndarray:
        return _euler_zyx_to_rotation(*self._ukf.x[self._ang])

    @property
    def x(self) -> np.ndarray:
        return self._ukf.x

    # ------------------------------------------------------------------
    # filterpy callbacks
    # ------------------------------------------------------------------

    def _fx(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Process model called by filterpy for each sigma point."""
        x_new = x.copy()

        roll, pitch, yaw = x[6], x[7], x[8]
        wx, wy, wz       = self._current_omega  # external input, not state

        # World-frame linear acceleration (GT override or IMU-derived)
        if self._gt_accel_world is not None:
            # Window already holds world-frame GT accel — use filtered value directly
            a_world = self._accel_filtered()
        else:
            # R = _euler_zyx_to_rotation(roll, pitch, yaw)
            # a_world = R @ self._accel_filtered()
            a_world = self._accel_filtered()
        # Position integration
        x_new[0:3] = x[0:3] + x[3:6] * dt

        # Velocity with exponential decay + IMU acceleration
        # decay = np.exp(-self.vel_decay_rate * dt)
        decay = 1
        x_new[3:6] = x[3:6] * decay + a_world * dt

        # Euler angle integration from IMU angular velocity
        cr, sr = np.cos(roll), np.sin(roll)
        cp = np.cos(pitch)
        tp = np.tan(pitch) if abs(cp) > 1e-6 else np.sign(np.sin(pitch)) * 1e6

        x_new[6] = _wrap_angle(roll  + (wx + (wy * sr + wz * cr) * tp) * dt)
        x_new[7] = _wrap_angle(pitch + (wy * cr - wz * sr) * dt)
        x_new[8] = _wrap_angle(yaw   + (wy * sr + wz * cr) / max(abs(cp), 1e-6) * np.sign(cp) * dt)

        return x_new

    def _hx_pos(self, x: np.ndarray) -> np.ndarray:
        return x[self._pos]

    def _hx_ang(self, x: np.ndarray) -> np.ndarray:
        return x[self._ang]
