import numpy as np


class HeightEstimator:
    """
    Estimates metric height by double-integrating IMU Z-acceleration (gravity removed).
    Uses the height change to compute a scale factor for monocular VO translation estimates.

    The ETH DAVIS IMU reports specific force: a_measured = a_kinematic + R_body_world.T @ g_world
    where g_world = [0, 0, -9.81] (gravity points down, world Z is up).
    So kinematic world-frame acceleration = R_world @ a_measured - [0, 0, g].
    """

    def __init__(self, g: float = 9.81, scale_ema_alpha: float = 0.1):
        self.g = g
        self.alpha = scale_ema_alpha  # EMA weight for scale updates

        self.height: float = 0.0
        self.velocity_z: float = 0.0
        self.last_height: float = 0.0
        self.scale: float = 1.0  # current best estimate of metric scale

        self.imu_updates: int = 0

    def update_imu(self, a_body: np.ndarray, R_world: np.ndarray, dt: float) -> None:
        """
        Integrate IMU linear acceleration to update height estimate.

        Parameters
        ----------
        a_body : (3,) array
            Linear acceleration in body/IMU frame [ax, ay, az] (m/s²).
        R_world : (3, 3) array
            Rotation matrix from body frame to world frame (R_world @ v_body = v_world).
        dt : float
            Time step in seconds.
        """
        if dt <= 0 or dt > 0.5:
            return

        # Transform body acceleration to world frame
        a_world = R_world @ a_body  # shape (3,)

        # Remove gravity (IMU measures a_kinematic + gravity reaction, so subtract g along world-Z)
        az_kinematic = a_world[2] - self.g

        # Integrate to get velocity and height
        self.velocity_z += az_kinematic * dt
        self.height += self.velocity_z * dt
        self.imu_updates += 1

    def compute_scale(self, visual_dz: float, min_motion_threshold: float = 0.005) -> float:
        """
        Compute/update metric scale by comparing IMU height change vs visual Z-displacement.

        Parameters
        ----------
        visual_dz : float
            Z-component of the unit translation vector from essential matrix decomposition.
        min_motion_threshold : float
            Minimum displacement (in respective units) to attempt scale estimation.

        Returns
        -------
        float
            Current best scale estimate (EMA-filtered).
        """
        imu_dz = self.height - self.last_height
        self.last_height = self.height

        # Only update scale when both signals show meaningful motion
        if abs(visual_dz) > min_motion_threshold and abs(imu_dz) > min_motion_threshold:
            if visual_dz * imu_dz > 0:  # same sign — consistent direction
                raw_scale = imu_dz / visual_dz
                raw_scale = float(np.clip(raw_scale, 0.01, 100.0))
                # Exponential moving average
                self.scale = (1.0 - self.alpha) * self.scale + self.alpha * raw_scale

        return self.scale

    def get_height(self) -> float:
        """Return current cumulative height estimate (m), starting from 0."""
        return self.height

    def reset(self) -> None:
        self.height = 0.0
        self.velocity_z = 0.0
        self.last_height = 0.0
        self.imu_updates = 0
