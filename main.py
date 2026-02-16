from src.VOEstimator import VOEstimator
import numpy as np
from src.Plotter import plot_trajectory_with_time_slider

def plot_estimated_trajectory(estimated_positions, estimated_timestamps, gt_positions=None, gt_timestamps=None):
    """
    Plot estimated trajectory with optional ground truth comparison.
    
    Parameters:
    -----------
    estimated_positions : array-like, shape (N, 2) or (N, 3)
        Estimated positions from your VO algorithm
    estimated_timestamps : array-like, optional
        Timestamps for estimated positions. If None, will use sequential indices.
    gt_csv_path : str, optional
        Path to ground truth CSV file
    """
    
    # Handle timestamps
    if estimated_timestamps is None:
        estimated_timestamps = np.arange(len(estimated_positions))
    
    estimated_timestamps = np.array(estimated_timestamps)
    estimated_positions = np.array(estimated_positions)

    # Handle ground truth data
    if gt_positions is not None and len(gt_positions) > 0:
        gt_positions = np.array(gt_positions)
        gt_timestamps = np.array(gt_timestamps)
        print(f"Ground truth: {len(gt_positions)} points available for plotting")
    else:
        gt_positions = None
        gt_timestamps = None
        print("No ground truth data available for plotting")

    # Determine if 2D or 3D
    plot_3d = estimated_positions.shape[1] >= 3
    
    # Create interactive plot
    plot_trajectory_with_time_slider(
        timestamps=estimated_timestamps,
        positions=estimated_positions,
        gt_timestamps=gt_timestamps,
        gt_positions=gt_positions,
        trajectory_label="Estimated Trajectory",
        gt_label="Ground Truth",
        plot_3d=plot_3d
    )
    # plot_trajectory_with_time_slider(
    #     timestamps=gt_timestamps,
    #     positions=gt_positions,
    #     trajectory_label="Estimated Trajectory",
    #     plot_3d=plot_3d
    # )

if __name__ == "__main__":
    dataset_file = r"data\outdoor_forward\outdoor_forward_1_davis_with_gt"
    calib_file = r"data\outdoor_forward_calib_davis"
    duration_seconds = None  # Process only first N seconds
    start_time = 1540113194.61788

    vo_estimator = VOEstimator(dataset_file, calib_file, duration=duration_seconds, use_imu=False, start_time=start_time)
    
    # Print time alignment information
    vo_estimator.data_manager.print_time_alignment_info()
    
    # Run VO estimation
    estimated_timestamps, estimated_positions = vo_estimator.run()
 
    # Get ground truth
    gt_timestamps, gt_positions, _ = vo_estimator.data_manager.get_groundtruth_trajectory()
    
    # Debug output
    print(f"\nEstimated trajectory: {len(estimated_positions)} points")
    print(f"Ground truth trajectory: {len(gt_positions)} points")
    if len(gt_positions) > 0:
        print(f"GT position range: X=[{gt_positions[:, 0].min():.3f}, {gt_positions[:, 0].max():.3f}], "
              f"Y=[{gt_positions[:, 1].min():.3f}, {gt_positions[:, 1].max():.3f}], "
              f"Z=[{gt_positions[:, 2].min():.3f}, {gt_positions[:, 2].max():.3f}]")
        print(f"GT timestamp range: [{gt_timestamps.min():.6f}, {gt_timestamps.max():.6f}]")
    
    plot_estimated_trajectory(estimated_positions, estimated_timestamps, gt_positions, gt_timestamps)
