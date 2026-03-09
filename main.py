from src.VOEstimator import VOEstimator
import numpy as np
from src.visualization import plot_trajectory_with_time_slider
from src.visualization import compute_ate, compute_rpe, compute_angle_error, print_metrics_table


def plot_estimated_trajectory(
    ukf_timestamps, ukf_positions, ukf_velocities, ukf_euler,
    gt_timestamps=None, gt_positions=None, gt_orientations=None,
    frames=None, imu_timestamps=None, imu_accel=None,
):
    """Interactive trajectory plot: UKF (IMU+VO fused) vs Ground Truth."""
    ukf_timestamps = np.array(ukf_timestamps)
    ukf_positions  = np.array(ukf_positions)

    if gt_positions is not None and len(gt_positions) > 0:
        gt_positions  = np.array(gt_positions)
        gt_timestamps = np.array(gt_timestamps)
        print(f"Ground truth: {len(gt_positions)} points available for plotting")
    else:
        gt_positions = gt_timestamps = None
        print("No ground truth data available for plotting")

    ukf_euler_arr = np.array(ukf_euler) if ukf_euler is not None and len(ukf_euler) > 0 else None
    ukf_vel_arr   = np.array(ukf_velocities) if ukf_velocities is not None and len(ukf_velocities) > 0 else None
    plot_3d = ukf_positions.shape[1] >= 3

    plot_trajectory_with_time_slider(
        timestamps=ukf_timestamps,
        positions=ukf_positions,
        orientations=ukf_euler_arr,
        velocities=ukf_vel_arr,
        gt_timestamps=gt_timestamps,
        gt_positions=gt_positions,
        gt_orientations=gt_orientations,
        trajectory_label="UKF (IMU+VO)",
        gt_label="Ground Truth",
        plot_3d=plot_3d,
        frames=frames,
        imu_timestamps=imu_timestamps,
        imu_accel=imu_accel,
    )


if __name__ == "__main__":
    dataset_file = r"data\outdoor_forward_1_snapdragon_with_gt"
    calib_file   = r"data\outdoor_forward_calib_davis"

    vo_estimator = VOEstimator(dataset_file, calib_file)
    vo_estimator.data_manager.print_time_alignment_info()

    # Run — UKF predicts from IMU at full rate; VO fused when use_vo=True
    (ukf_timestamps, ukf_positions, ukf_velocities, ukf_euler,
     gt_frame_timestamps, gt_frame_positions, gt_frame_orientations) = vo_estimator.run()

    # Full-resolution GT for metrics
    gt_timestamps, gt_positions, gt_orientations = (
        vo_estimator.data_manager.get_groundtruth_trajectory()
    )

    print(f"\nUKF trajectory:       {len(ukf_positions)} poses")
    print(f"Ground truth (CSV):   {len(gt_positions)} poses")
    print(f"GT at image frames:   {len(gt_frame_positions)} poses (aligned)")

    # -------------------------------------------------------
    # Trajectory metrics
    # -------------------------------------------------------
    print("\nComputing trajectory metrics...")

    ukf_ate = ukf_rpe = ukf_angle_err = {}
    if len(ukf_positions) > 1:
        ukf_ts_arr  = np.array(ukf_timestamps)
        gt_ts_arr   = gt_timestamps
        gt_pos_arr  = gt_positions
        gt_ori_arr  = gt_orientations   # [x,y,z,w] quaternions
        ukf_ori_dummy = np.tile(np.eye(3), (len(ukf_positions), 1, 1))

        ukf_ate = compute_ate(ukf_ts_arr, ukf_positions, ukf_ori_dummy,
                              gt_ts_arr, gt_pos_arr, gt_ori_arr)
        ukf_rpe = compute_rpe(ukf_ts_arr, ukf_positions, ukf_ori_dummy,
                              gt_ts_arr, gt_pos_arr, gt_ori_arr,
                              delta=1, delta_unit="f")
        if ukf_euler is not None and len(ukf_euler) > 1:
            ukf_angle_err = compute_angle_error(ukf_ts_arr, ukf_euler,
                                                gt_ts_arr, gt_ori_arr)

    print_metrics_table({}, {}, ukf_ate, ukf_rpe, ukf_angle_err)

    # -------------------------------------------------------
    # GT height offset for plotting (align with UKF start)
    # -------------------------------------------------------
    gt_positions_plot    = gt_frame_positions.copy() if len(gt_frame_positions) > 0 else gt_frame_positions
    gt_timestamps_plot   = np.array(gt_frame_timestamps)
    gt_orientations_plot = gt_frame_orientations
    if (len(gt_positions_plot) > 0 and len(ukf_positions) > 0
            and gt_positions_plot.shape[1] >= 3):
        gt_z_offset = float(ukf_positions[0, 2]) - float(gt_positions_plot[0, 2])
        if abs(gt_z_offset) > 1e-4:
            print(f"GT Z offset for plotting: {gt_z_offset:+.4f} m")
            gt_positions_plot = gt_positions_plot.copy()
            gt_positions_plot[:, 2] += gt_z_offset

    # -------------------------------------------------------
    # Interactive plot
    # -------------------------------------------------------
    imu_timestamps_arr = np.array(vo_estimator._imu_ts) if vo_estimator._imu_ts else None
    imu_accel_arr      = np.array(vo_estimator._imu_acc) if vo_estimator._imu_acc else None

    plot_estimated_trajectory(
        ukf_timestamps, ukf_positions, ukf_velocities, ukf_euler,
        gt_timestamps=gt_timestamps_plot,
        gt_positions=gt_positions_plot,
        gt_orientations=gt_orientations_plot,
        frames=vo_estimator.frames,
        imu_timestamps=imu_timestamps_arr,
        imu_accel=imu_accel_arr,
    )
