# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
# Activate the virtual environment first
.venv/Scripts/activate

# Run the full VO pipeline
python main.py
```

The entry point is `main.py`. Dataset path and calibration path are hardcoded at the bottom. Runtime parameters are loaded from `data/outdoor_forward/outdoor_forward_1_davis_with_gt/params.yml`.

## params.yml controls

| Key | Effect |
|-----|--------|
| `use_imu` | Process IMU events (angular velocity for orientation, linear acceleration for UKF/height) |
| `start_time` | Absolute timestamp to begin processing |
| `duration` | Seconds of data to process (capped by GT end time) |
| `show_frames` | Open an OpenCV window showing tracked features during processing |
| `matcher` | Feature matcher: `sift` (default), `orb`, `lk`, `lightglue`, `loftr` |

## Architecture overview

This is a monocular Visual Odometry system for the ETH DAVIS event camera dataset. Data flows as:

```
DataManager (time-sorted CSV events)
    → VOEstimator.run() loop
        → update_attitude(imu_event)   # IMU: Rodrigues integration → imu_orientation
                                       #      linear accel → HeightEstimator + UKF.predict()
        → update(image_frame)          # Image: undistort → feature match → estimate_pose
                                       #        → UKF.update(cumulative_position)
    → calculate_trajectory()           # returns VO trajectory + UKF trajectory
    → compute_ate() / compute_rpe()    # prints metrics table
    → plot_estimated_trajectory()      # 3-line interactive plot
```

### Module responsibilities

- **`src/DataManager.py`** — Reads three CSVs (`dvs-image_raw.csv`, `dvs-imu.csv`, `groundtruth-pose.csv`), merges into a time-sorted event list, exposes a Python iterator. Images decoded lazily via `DavisCsvReader`.
- **`src/DavisCsvReader.py`** — Decodes `mono8`/`8uc1` images stored as Python byte-string literals in CSV rows.
- **`src/ConfigManager.py`** — Loads the three calibration YAMLs. Provides camera matrix K (3×3), equidistant (fisheye) distortion coeffs, T_cam_imu (4×4), IMU noise params.
- **`src/FeaturesFlow.py`** — Feature detection and matching. Active matchers: `SIFT_feature_matching` (default CPU), `LightGlue_matching` (GPU, SuperPoint keypoints), `LOFTR_matching` (GPU, detector-free via kornia). Selector: `get_matcher(name)`. GPU models are cached after first call.
- **`src/HeightEstimator.py`** — Double-integrates IMU Z-acceleration (gravity removed) to track metric height. `compute_scale(visual_dz)` returns an EMA-filtered scale factor that replaces the old `/10` hack in `VOEstimator`.
- **`src/UKF.py`** — 6-state `[x,y,z,vx,vy,vz]` Unscented Kalman Filter. `predict(accel_world, dt)` uses IMU linear acceleration; `update(position)` fuses VO position measurements. Orientation is not part of the UKF state — it is handled externally by IMU integration.
- **`src/Metrics.py`** — ATE and RPE metrics via the `evo` library (already installed). Quaternion format conversion handled internally. `print_metrics_table(vo_ate, vo_rpe, ukf_ate, ukf_rpe)` prints a side-by-side comparison.
- **`src/VOEstimator.py`** — Main VO class. `estimate_pose()` uses `HeightEstimator.compute_scale()` for metric scale (no more `/10`). `run()` now returns `(vo_timestamps, vo_positions, vo_orientations, ukf_timestamps, ukf_positions)`.
- **`src/Plotter.py`** — Interactive matplotlib plot with time slider. Accepts optional `ukf_timestamps`/`ukf_positions` for a third orange trajectory line. Also contains an Open3D `TrajectoryViewer` (module-level executable code at the bottom — runs only when `Plotter.py` is executed directly).

### Dataset / calibration layout

```
data/
  outdoor_forward_calib_davis/
    camchain-imucam-..outdoor_forward_calib_davis_imu.yaml  # camera intrinsics, T_cam_imu
    imu-..outdoor_forward_calib_davis_imu.yaml              # IMU noise params
    target.yaml
  outdoor_forward/
    outdoor_forward_1_davis_with_gt/
      dvs-image_raw.csv        # frames as byte strings
      dvs-imu.csv              # angular_velocity.x/y/z + linear_acceleration.x/y/z
      groundtruth-pose.csv     # 6-DOF poses (position + quaternion [x,y,z,w])
      params.yml               # runtime parameters (see table above)
```

### IMU data format

The IMU CSV exposes every column via `row.to_dict()` in `DataManager._build_event_list()`. Columns accessed in code:
- `angular_velocity.x/y/z` — used by `update_attitude()` for orientation integration
- `linear_acceleration.x/y/z` — used by `HeightEstimator` and `UKF.predict()`

### Known remaining issues

- Scale: `HeightEstimator` uses double-integrated IMU acceleration which drifts over time; for long sequences, consider periodic GT realignment or a barometer.
- Rotation accumulation: `R` from `cv2.recoverPose` is used only to update `current_orientation`; the trajectory is still accumulated via translation cumsum rather than a proper pose chain.
- UKF is position-only (6 states); adding orientation to the state and proper IMU pre-integration would improve accuracy.
