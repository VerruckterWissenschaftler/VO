import numpy as np
import logging

logger = logging.getLogger("Metrics")


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _orientations_to_quat_wxyz(orientations: np.ndarray) -> np.ndarray:
    """
    Convert orientation array to [w, x, y, z] quaternions.
    Accepts:
      - (N, 3, 3) rotation matrices
      - (N, 4)    quaternions in [x, y, z, w]  (GT format from DataManager)
      - (N, 4)    quaternions in [w, x, y, z]
    Returns (N, 4) in [w, x, y, z] order (evo convention).
    """
    if orientations.ndim == 3 and orientations.shape[-2:] == (3, 3):
        return np.array([_rotation_matrix_to_quat_wxyz(R) for R in orientations])

    if orientations.ndim == 2 and orientations.shape[1] == 4:
        # Heuristic: if w component (last column) has values ≥ other components on average,
        # it's likely [x,y,z,w] (DataManager format) → reorder to [w,x,y,z]
        if np.mean(np.abs(orientations[:, 3])) >= np.mean(np.abs(orientations[:, 0])):
            return orientations[:, [3, 0, 1, 2]]  # [x,y,z,w] → [w,x,y,z]
        return orientations  # already [w,x,y,z]

    raise ValueError(f"Unsupported orientations shape: {orientations.shape}")


def compute_ate(
    est_timestamps: np.ndarray,
    est_positions: np.ndarray,
    est_orientations: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_orientations: np.ndarray,
    correct_scale: bool = False,
) -> dict:
    """
    Compute Absolute Trajectory Error (ATE) using the evo library.

    Parameters
    ----------
    est_timestamps : (N,) timestamps in seconds
    est_positions  : (N, 3) estimated [x, y, z]
    est_orientations : (N, 3, 3) rotation matrices or (N, 4) quaternions
    gt_timestamps  : (M,) ground-truth timestamps
    gt_positions   : (M, 3) ground-truth [x, y, z]
    gt_orientations : (M, 4) quaternions [x,y,z,w] (DataManager format)
    correct_scale  : bool — whether to apply scale correction during alignment

    Returns
    -------
    dict with keys: rmse, mean, median, std, min, max  (all in metres)
    """
    try:
        from evo.core.trajectory import PoseTrajectory3D
        from evo.core import sync, metrics
    except ImportError:
        logger.error("evo library not found. Install with: pip install evo")
        return {}

    if len(est_positions) < 2 or len(gt_positions) < 2:
        logger.warning("Not enough poses for ATE computation.")
        return {}

    est_quat = _orientations_to_quat_wxyz(est_orientations)
    gt_quat = _orientations_to_quat_wxyz(gt_orientations)

    traj_est = PoseTrajectory3D(
        positions_xyz=est_positions,
        orientations_quat_wxyz=est_quat,
        timestamps=est_timestamps,
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_positions,
        orientations_quat_wxyz=gt_quat,
        timestamps=gt_timestamps,
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    try:
        traj_est.align(traj_ref, correct_scale=correct_scale)
    except Exception as e:  # GeometryException (degenerate trajectory)
        logger.warning("Trajectory alignment failed (%s) — skipping ATE.", e)
        return {}

    ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ate_metric.process_data((traj_ref, traj_est))
    return ate_metric.get_all_statistics()


def compute_rpe(
    est_timestamps: np.ndarray,
    est_positions: np.ndarray,
    est_orientations: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_orientations: np.ndarray,
    delta: int = 1,
    delta_unit: str = "f",
) -> dict:
    """
    Compute Relative Pose Error (RPE) using the evo library.

    Parameters
    ----------
    delta      : interval between pose pairs (integer for frames/index)
    delta_unit : 'f' for frames (default), 'm' for meters, 'deg'/'rad' for rotation
                 Note: evo RPE does NOT support time-based ('s') delta units.

    Returns
    -------
    dict with keys: rmse, mean, median, std, min, max  (metres)
    """
    try:
        from evo.core.trajectory import PoseTrajectory3D
        from evo.core import sync, metrics
    except ImportError:
        logger.error("evo library not found.")
        return {}

    if len(est_positions) < 2 or len(gt_positions) < 2:
        logger.warning("Not enough poses for RPE computation.")
        return {}

    _unit_map = {
        "f": metrics.Unit.frames,
        "m": metrics.Unit.meters,
        "deg": metrics.Unit.degrees,
        "rad": metrics.Unit.radians,
    }
    unit = _unit_map.get(delta_unit, metrics.Unit.frames)

    est_quat = _orientations_to_quat_wxyz(est_orientations)
    gt_quat = _orientations_to_quat_wxyz(gt_orientations)

    traj_est = PoseTrajectory3D(
        positions_xyz=est_positions,
        orientations_quat_wxyz=est_quat,
        timestamps=est_timestamps,
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_positions,
        orientations_quat_wxyz=gt_quat,
        timestamps=gt_timestamps,
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    rpe_metric = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=unit,
        all_pairs=False,
    )
    rpe_metric.process_data((traj_ref, traj_est))
    return rpe_metric.get_all_statistics()


def compute_angle_error(
    est_timestamps: np.ndarray,
    est_euler_rad: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_orientations: np.ndarray,
) -> dict:
    """
    Compute RMSE and mean angle error between estimated euler angles and GT quaternions.

    Parameters
    ----------
    est_timestamps : (N,) seconds
    est_euler_rad  : (N, 3) [roll, pitch, yaw] in radians
    gt_timestamps  : (M,) seconds
    gt_orientations: (M, 4) quaternions [x,y,z,w]

    Returns
    -------
    dict with keys: rmse, mean  (degrees)
    """
    if len(est_euler_rad) < 2 or len(gt_orientations) < 2:
        return {}

    def _quat_xyzw_to_euler(q):
        """[x,y,z,w] → [roll, pitch, yaw] in radians."""
        x, y, z, w = q
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        sinp  = 2*(w*y - z*x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return np.array([roll, pitch, yaw])

    errors = []
    for i, t in enumerate(est_timestamps):
        gt_idx = int(np.argmin(np.abs(gt_timestamps - t)))
        gt_euler = _quat_xyzw_to_euler(gt_orientations[gt_idx])
        diff = est_euler_rad[i] - gt_euler
        # Wrap each angle diff to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        errors.append(np.linalg.norm(np.degrees(diff)))

    errors = np.array(errors)
    return {"rmse": float(np.sqrt(np.mean(errors**2))), "mean": float(np.mean(errors))}


def print_metrics_table(
    vo_ate: dict, vo_rpe: dict,
    ukf_ate: dict | None = None, ukf_rpe: dict | None = None,
    ukf_angle_error: dict | None = None,
) -> None:
    """Print a formatted comparison table of ATE, RPE and angle error metrics."""
    has_ukf = bool(ukf_ate or ukf_rpe or ukf_angle_error)
    header = f"{'Metric':<22} {'VO':>10}"
    if has_ukf:
        header += f" {'UKF':>10}"
    print("\n" + "=" * (len(header) + 2))
    print(" Trajectory Evaluation Metrics")
    print("=" * (len(header) + 2))
    print(header)
    print("-" * (len(header) + 2))

    def _row(name, vo_val, ukf_val=None):
        line = f"  {name:<20} {vo_val:>10.4f}"
        if ukf_val is not None:
            line += f" {ukf_val:>10.4f}"
        print(line)

    def _nan():
        return float("nan")

    if vo_ate or ukf_ate:
        print("  ATE (m):")
        _row("  RMSE",   vo_ate.get("rmse",   _nan()) if vo_ate else _nan(), ukf_ate.get("rmse",   _nan()) if ukf_ate else None)
        _row("  Mean",   vo_ate.get("mean",   _nan()) if vo_ate else _nan(), ukf_ate.get("mean",   _nan()) if ukf_ate else None)
        _row("  Median", vo_ate.get("median", _nan()) if vo_ate else _nan(), ukf_ate.get("median", _nan()) if ukf_ate else None)
        _row("  Std",    vo_ate.get("std",    _nan()) if vo_ate else _nan(), ukf_ate.get("std",    _nan()) if ukf_ate else None)

    if vo_rpe or ukf_rpe:
        print("  RPE (m, delta=1 frame):")
        _row("  RMSE",  vo_rpe.get("rmse", _nan()) if vo_rpe else _nan(), ukf_rpe.get("rmse", _nan()) if ukf_rpe else None)
        _row("  Mean",  vo_rpe.get("mean", _nan()) if vo_rpe else _nan(), ukf_rpe.get("mean", _nan()) if ukf_rpe else None)

    if ukf_angle_error:
        print("  Angle Error (°):")
        _row("  RMSE", _nan(), ukf_angle_error.get("rmse", _nan()))
        _row("  Mean", _nan(), ukf_angle_error.get("mean", _nan()))

    print("=" * (len(header) + 2) + "\n")
