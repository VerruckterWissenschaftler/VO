from src.VOEstimator import VOEstimator
import numpy as np
from src.Plotter import plot_trajectory_with_time_slider
from src.Metrics import compute_ate, compute_rpe, print_metrics_table


def plot_estimated_trajectory(
    vo_timestamps, vo_positions, vo_orientations,
    gt_timestamps=None, gt_positions=None, gt_orientations=None,
    ukf_timestamps=None, ukf_positions=None,
):
    """
    Interactive trajectory plot with three lines:
      - Blue:   Raw VO estimate
      - Orange: UKF-filtered estimate
      - Green:  Ground truth
    """
    if vo_timestamps is None:
        vo_timestamps = np.arange(len(vo_positions))
    vo_timestamps = np.array(vo_timestamps)
    vo_positions = np.array(vo_positions)

    if gt_positions is not None and len(gt_positions) > 0:
        gt_positions = np.array(gt_positions)
        gt_timestamps = np.array(gt_timestamps)
        print(f"Ground truth: {len(gt_positions)} points available for plotting")
    else:
        gt_positions = gt_timestamps = None
        print("No ground truth data available for plotting")

    ukf_pos_arr = np.array(ukf_positions) if ukf_positions is not None and len(ukf_positions) > 0 else None
    ukf_ts_arr = np.array(ukf_timestamps) if ukf_timestamps is not None and len(ukf_timestamps) > 0 else None

    plot_3d = vo_positions.shape[1] >= 3

    plot_trajectory_with_time_slider(
        timestamps=vo_timestamps,
        positions=vo_positions,
        orientations=vo_orientations,
        gt_timestamps=gt_timestamps,
        gt_positions=gt_positions,
        gt_orientations=gt_orientations,
        trajectory_label="VO Estimate",
        gt_label="Ground Truth",
        ukf_timestamps=ukf_ts_arr,
        ukf_positions=ukf_pos_arr,
        ukf_label="UKF Filtered",
        plot_3d=plot_3d,
    )


if __name__ == "__main__":
    dataset_file = r"data\outdoor_forward\outdoor_forward_1_davis_with_gt"
    calib_file = r"data\outdoor_forward_calib_davis"

    vo_estimator = VOEstimator(dataset_file, calib_file)
    vo_estimator.data_manager.print_time_alignment_info()

    # Run VO — returns raw VO trajectory and UKF-filtered trajectory
    (vo_timestamps, vo_positions, vo_orientations,
     ukf_timestamps, ukf_positions) = vo_estimator.run()

    # Ground truth
    gt_timestamps, gt_positions, gt_orientations = (
        vo_estimator.data_manager.get_groundtruth_trajectory()
    )

    print(f"\nEstimated trajectory (VO):  {len(vo_positions)} poses")
    print(f"UKF filtered trajectory:    {len(ukf_positions)} poses")
    print(f"Ground truth trajectory:    {len(gt_positions)} poses")

    # -------------------------------------------------------
    # Trajectory metrics
    # -------------------------------------------------------
    vo_ts_arr = np.array(vo_timestamps)
    gt_ts_arr = gt_timestamps
    gt_pos_arr = gt_positions
    gt_ori_arr = gt_orientations  # [x,y,z,w] quaternions from DataManager

    print("\nComputing trajectory metrics...")

    vo_ate = compute_ate(vo_ts_arr, vo_positions, vo_orientations,
                         gt_ts_arr, gt_pos_arr, gt_ori_arr)
    vo_rpe = compute_rpe(vo_ts_arr, vo_positions, vo_orientations,
                         gt_ts_arr, gt_pos_arr, gt_ori_arr,
                         delta=1, delta_unit="f")

    ukf_ate, ukf_rpe = {}, {}
    if len(ukf_positions) > 1:
        # UKF uses identity orientations (orientation tracked externally)
        ukf_ori = np.tile(np.eye(3), (len(ukf_positions), 1, 1))
        ukf_ts_arr = np.array(ukf_timestamps)
        ukf_ate = compute_ate(ukf_ts_arr, ukf_positions, ukf_ori,
                              gt_ts_arr, gt_pos_arr, gt_ori_arr)
        ukf_rpe = compute_rpe(ukf_ts_arr, ukf_positions, ukf_ori,
                              gt_ts_arr, gt_pos_arr, gt_ori_arr,
                              delta=1, delta_unit="f")

    print_metrics_table(vo_ate, vo_rpe, ukf_ate, ukf_rpe)

    # -------------------------------------------------------
    # GT height offset for plotting only
    # (VO starts at init_height; shift GT Z so both start at the same value)
    # -------------------------------------------------------
    gt_positions_plot = gt_positions.copy() if gt_positions is not None and len(gt_positions) > 0 else gt_positions
    if (gt_positions_plot is not None and len(gt_positions_plot) > 0
            and len(vo_positions) > 0
            and gt_positions_plot.shape[1] >= 3):
        gt_z_offset = float(vo_positions[0, 2]) - float(gt_positions_plot[0, 2])
        if abs(gt_z_offset) > 1e-4:
            print(f"GT Z offset for plotting: {gt_z_offset:+.4f} m  "
                  f"(VO init Z={vo_positions[0, 2]:.4f}, GT init Z={gt_positions_plot[0, 2]:.4f})")
            gt_positions_plot = gt_positions_plot.copy()
            gt_positions_plot[:, 2] += gt_z_offset

    # -------------------------------------------------------
    # Interactive plot
    # -------------------------------------------------------
    plot_estimated_trajectory(
        vo_timestamps, vo_positions, vo_orientations,
        gt_positions=gt_positions_plot,
        gt_timestamps=gt_timestamps,
        gt_orientations=gt_orientations,
        ukf_timestamps=ukf_timestamps,
        ukf_positions=ukf_positions,
    )
