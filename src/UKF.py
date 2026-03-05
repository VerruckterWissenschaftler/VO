import numpy as np


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _make_spd(P: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Project P to the nearest symmetric positive-definite matrix."""
    P = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


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


class UKF:
    """
    6-state UKF for per-frame shift (translation delta) filtering.

    State vector: x = [vx, vy, vz, roll, pitch, yaw]

    The UKF estimates camera velocity (world frame) and orientation.
    Absolute position is accumulated externally as a cumulative sum of
    filtered per-frame shifts.

    Predict: driven by body-frame gyro (orientation) + body-frame linear
             acceleration (velocity). Gravity is removed internally using
             the current sigma-point orientation and g_world = 9.81 m/s².
    Update:  VO-derived per-frame shift measurement:  z_hat = v * dt.
    ZUPT:    Zero-velocity update during static periods.
    """

    _G = 9.81  # gravitational acceleration, m/s²

    def __init__(
        self,
        process_noise_vel:   float = 1e-4,   # (m/s)² per step — tuned to ~200 Hz IMU
        process_noise_angle: float = 1e-5,   # rad² per step
        measurement_noise:   float = 0.5,
        initial_uncertainty: float = 1e-3,
        vel_decay_rate:      float = 2.0,    # 1/s — velocity e-folding time = 0.5 s
    ):
        self.n = 6   # state dimension: [vx, vy, vz, roll, pitch, yaw]
        self.m = 3   # measurement dimension: shift [dx, dy, dz]

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * initial_uncertainty

        self.vel_decay_rate = vel_decay_rate

        self.Q = np.diag([
            process_noise_vel,   process_noise_vel,   process_noise_vel,
            process_noise_angle, process_noise_angle, process_noise_angle,
        ])

        # Shift measurement noise (in metres)
        self.R_meas = np.eye(self.m) * measurement_noise

        # UKF scaling (Wan & van der Merwe 2000)
        self.alpha = 1e-3
        self.beta  = 2.0
        self.kappa = 0.0
        self._lam  = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.Wm, self.Wc = self._compute_weights()

        self.initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, R: np.ndarray | None = None) -> None:
        """Set initial orientation; velocity starts at zero."""
        self.x[:3] = 0.0
        self.x[3:] = _rotation_to_euler_zyx(R) if R is not None else 0.0
        self.initialized = True

    def predict(self, omega: np.ndarray, accel_body: np.ndarray, dt: float) -> None:
        """
        Prediction step driven by body-frame gyro and linear acceleration.

        Parameters
        ----------
        omega      : (3,) ndarray  — angular velocity [wx, wy, wz] in body frame, rad/s.
        accel_body : (3,) ndarray  — linear acceleration [ax, ay, az] in body frame, m/s².
                                     Gravity is removed internally per sigma-point orientation.
        dt         : float         — time step in seconds.
        """
        if not self.initialized or dt <= 0 or dt > 0.5:
            return

        sigma_pts = self._sigma_points(self.x, self.P)

        propagated = np.array([
            self._process_model(sp, omega, accel_body, dt, self._G)
            for sp in sigma_pts
        ])

        x_pred = self._weighted_mean(propagated)
        P_pred = self.Q.copy()
        for i, sp in enumerate(propagated):
            diff = sp - x_pred
            diff[3:] = np.array([_wrap_angle(d) for d in diff[3:]])
            P_pred += self.Wc[i] * np.outer(diff, diff)

        self.x = x_pred
        self.P = _make_spd(P_pred)

    def update(self, vo_shift: np.ndarray, dt: float) -> None:
        """
        Update step with a VO-derived per-frame shift measurement.

        Measurement model:  z_hat = v * dt
        Innovation:         vo_shift - v_pred * dt

        Parameters
        ----------
        vo_shift : (3,) ndarray — measured translation [dx, dy, dz] in world frame.
        dt       : float        — time elapsed since last frame (seconds).
        """
        if dt <= 0:
            return

        if not self.initialized:
            # Bootstrap velocity from first shift observation
            self.x[:3] = vo_shift / dt
            self.initialized = True
            return

        sigma_pts = self._sigma_points(self.x, self.P)

        # Predicted measurement: h(x) = v * dt
        z_sigma = np.array([sp[:3] * dt for sp in sigma_pts])
        z_pred  = z_sigma.T @ self.Wm

        S   = self.R_meas.copy()
        Pxz = np.zeros((self.n, self.m))
        for i, (sp, zsp) in enumerate(zip(sigma_pts, z_sigma)):
            dz = zsp - z_pred
            dx = sp - self.x
            dx[3:] = np.array([_wrap_angle(d) for d in dx[3:]])
            S   += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (vo_shift - z_pred)
        self.x[3:] = np.array([_wrap_angle(a) for a in self.x[3:]])
        self.P = _make_spd(self.P - K @ S @ K.T)

    def zupt_update(self) -> None:
        """Zero-Velocity Update: constrain velocity to zero (static camera)."""
        if not self.initialized:
            return

        sigma_pts = self._sigma_points(self.x, self.P)
        z_sigma   = np.array([sp[:3] for sp in sigma_pts])
        z_pred    = z_sigma.T @ self.Wm

        R_zupt = np.eye(3) * 1e-4   # tight noise → strong zero-velocity correction
        S   = R_zupt.copy()
        Pxz = np.zeros((self.n, 3))
        for i, (sp, zsp) in enumerate(zip(sigma_pts, z_sigma)):
            dz = zsp - z_pred
            dx = sp - self.x
            dx[3:] = np.array([_wrap_angle(d) for d in dx[3:]])
            S   += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (np.zeros(3) - z_pred)
        self.x[3:] = np.array([_wrap_angle(a) for a in self.x[3:]])
        self.P = _make_spd(self.P - K @ S @ K.T)

    def get_filtered_shift(self, dt: float) -> np.ndarray:
        """Return the filtered per-frame shift estimate: v * dt."""
        return self.x[:3] * dt

    def get_rotation_matrix(self) -> np.ndarray:
        """Return 3×3 rotation matrix from UKF Euler angle states."""
        return _euler_zyx_to_rotation(*self.x[3:])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(self):
        n, lam = self.n, self._lam
        num_pts = 2 * n + 1
        Wm = np.full(num_pts, 0.5 / (n + lam))
        Wm[0] = lam / (n + lam)
        Wc = Wm.copy()
        Wc[0] += 1.0 - self.alpha ** 2 + self.beta
        return Wm, Wc

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n, lam = self.n, self._lam
        L = np.linalg.cholesky(_make_spd((n + lam) * P))

        pts = np.empty((2 * n + 1, n))
        pts[0] = x
        for i in range(n):
            pts[i + 1]     = x + L[:, i]
            pts[n + i + 1] = x - L[:, i]
        pts[:, 3:] = np.vectorize(_wrap_angle)(pts[:, 3:])
        return pts

    def _weighted_mean(self, propagated: np.ndarray) -> np.ndarray:
        """Weighted mean with circular mean for angle states (indices 3–5)."""
        x_mean = propagated.T @ self.Wm
        for i in range(3, 6):
            angles = propagated[:, i]
            x_mean[i] = np.arctan2(
                np.sum(self.Wm * np.sin(angles)),
                np.sum(self.Wm * np.cos(angles)),
            )
        return x_mean

    def _process_model(
        self, x: np.ndarray, omega: np.ndarray, accel_body: np.ndarray, dt: float, g: float
    ) -> np.ndarray:
        """
        Process model: velocity integrates from world-frame linear acceleration
        (gravity removed using sigma-point orientation); Euler angles integrate
        from body-frame gyro rates.
        """
        x_new = x.copy()

        roll, pitch, yaw = x[3], x[4], x[5]
        R = _euler_zyx_to_rotation(roll, pitch, yaw)

        # Rotate body-frame acceleration to world frame and remove gravity
        a_world = R @ accel_body
        a_world[2] -= g  # world Z is up; IMU measures a_kin + g_reaction

        # Velocity integration with exponential decay: v decays toward 0 when
        # no VO measurement corrects it, preventing unbounded drift from accel bias.
        decay = np.exp(-self.vel_decay_rate * dt)
        x_new[:3] = x[:3] * decay + a_world * dt

        # Euler angle update from body-frame angular velocity
        wx, wy, wz = omega
        cr, sr = np.cos(roll), np.sin(roll)
        cp = np.cos(pitch)
        tp = np.tan(pitch) if abs(cp) > 1e-6 else np.sign(np.sin(pitch)) * 1e6

        droll  = (wx + (wy * sr + wz * cr) * tp) * dt
        dpitch = (wy * cr - wz * sr) * dt
        dyaw   = (wy * sr + wz * cr) / max(abs(cp), 1e-6) * np.sign(cp) * dt

        x_new[3] = _wrap_angle(roll  + droll)
        x_new[4] = _wrap_angle(pitch + dpitch)
        x_new[5] = _wrap_angle(yaw   + dyaw)

        return x_new
