# Visual Odometry Pipeline Review

**Date:** February 16, 2026  
**Status:** Excluding IMU integration (work in progress)

---

## 🔴 Critical Issues

### 1. Scale Recovery is Arbitrary
**Location:** `src/VOEstimator.py:118`

```python
trajectory = np.cumsum(estimated_positions, axis=0) / 10 + self.current_pose
```

**Problem:** Dividing by 10 is completely arbitrary and has no physical meaning.

**Impact:** Trajectory scale is incorrect, making comparison with ground truth meaningless.

**Fix Options:**
- Implement proper scale alignment with ground truth for the first N meters
- Use IMU accelerometer data for metric scale recovery
- Compute scale factor from ground truth and apply it consistently

**Suggested Implementation:**
```python
def compute_scale_from_gt(self, estimated_deltas, gt_trajectory, num_frames=30):
    """Compute scale factor from ground truth over first N frames"""
    est_distance = np.sum(np.linalg.norm(estimated_deltas[:num_frames], axis=1))
    gt_distance = np.sum(np.linalg.norm(np.diff(gt_trajectory[:num_frames+1], axis=0), axis=1))
    return gt_distance / est_distance if est_distance > 0 else 1.0
```

---

### 2. Empty IMU Integration
**Location:** `src/VOEstimator.py:96-98`

```python
def update_attitude(self, imu_data):
    # self.R = imu_data
    pass
```

**Problem:** IMU data is collected but completely unused.

**Impact:** 
- Missing opportunity for better rotation estimation
- Missing metric scale from accelerometer
- Currently estimating 5-DOF motion (rotation + translation direction) but treating as 6-DOF

**Note:** Acknowledged as work in progress. IMU integration should provide:
- Better rotation estimation (reduce drift)
- Metric scale through double integration of acceleration
- Gravity direction for absolute orientation

---

### 3. Missing Rotation Accumulation
**Location:** `src/VOEstimator.py:85-86, 118`

**Problem:** You estimate rotation matrix `R` from `cv2.recoverPose` but never use it.

```python
def estimate_pose(self, p0_good, p1_good):
    E, mask, x0, x1 = self.derive_essential_matrix(p0_good, p1_good)
    _, R, t, mask_pose = cv2.recoverPose(E, x1, x0, focal=1.0, pp=(0, 0))
    t = np.array(t).flatten()
    return t  # <-- Only returning translation, R is thrown away!
```

**Impact:** Translation vectors are accumulated in the wrong reference frame, causing incorrect trajectory.

**Fix:** Maintain world rotation and transform each translation:
```python
def __init__(self, ...):
    # ...
    self.R_world = np.eye(3)  # World rotation matrix
    self.position = np.zeros(3)  # World position

def estimate_pose(self, p0_good, p1_good):
    E, mask, x0, x1 = self.derive_essential_matrix(p0_good, p1_good)
    _, R, t, mask_pose = cv2.recoverPose(E, x1, x0, focal=1.0, pp=(0, 0))
    t = np.array(t).flatten()
    return R, t  # <-- Return both!

def calculate_trajectory(self):
    estimated_data = [(t, self.shifts[t]) for t in sorted(self.shifts.keys())]
    
    position = self.current_pose.copy()
    R_world = np.eye(3)
    trajectory = [position.copy()]
    
    for timestamp, (R_rel, t_rel) in estimated_data[1:]:
        # Update world rotation
        R_world = R_world @ R_rel
        # Transform translation to world frame with scale
        position += R_world @ (self.scale * t_rel)
        trajectory.append(position.copy())
    
    timestamps = [t for t, _ in estimated_data]
    return np.array(timestamps), np.array(trajectory)
```

---

### 4. Trajectory Coordinate Frame Issues
**Location:** `src/VOEstimator.py:118`

**Problem:** Simple `np.cumsum` assumes all translation vectors are in the same coordinate frame.

**Reality:** Each translation `t` from `cv2.recoverPose` is in the LOCAL camera frame at that time step.

**Correct Approach:** 
1. Maintain accumulated rotation matrix `R_world`
2. For each new frame: `R_world = R_world @ R_relative`
3. Transform translation: `position = position + R_world @ (scale * t_local)`

---

## 🟡 Major Issues

### 5. Hard-Coded Feature Detection Parameters
**Location:** `src/FeaturesFlow.py:4-11`

```python
def detect_features(frame):
    p0 = cv2.goodFeaturesToTrack(
        frame,
        maxCorners=1000,      # Hard-coded
        qualityLevel=0.1,     # Hard-coded
        minDistance=5,        # Hard-coded
        blockSize=7           # Hard-coded
    )
    return p0
```

**Problem:** Parameters should be configurable for different environments.

**Suggestion:** Create a configuration system:
```python
class FeatureConfig:
    MAX_CORNERS = 1000
    QUALITY_LEVEL = 0.1
    MIN_DISTANCE = 5
    BLOCK_SIZE = 7

def detect_features(frame, config=FeatureConfig()):
    p0 = cv2.goodFeaturesToTrack(
        frame,
        maxCorners=config.MAX_CORNERS,
        qualityLevel=config.QUALITY_LEVEL,
        minDistance=config.MIN_DISTANCE,
        blockSize=config.BLOCK_SIZE
    )
    return p0
```

---

### 6. No Keyframe Selection Strategy
**Location:** `src/VOEstimator.py:114-129`

**Problem:** Processing every single frame leads to:
- Computational inefficiency
- Accumulated drift from small baselines
- Poor conditioning of essential matrix estimation

**Impact:** Small motion between consecutive frames causes numerical instability.

**Suggestion:** Implement keyframe selection:
```python
def should_create_keyframe(self, current_R, current_t, num_features):
    """Decide if current frame should be a keyframe"""
    # Rotation angle from trace of rotation matrix
    rotation_angle = np.arccos((np.trace(current_R) - 1) / 2)
    translation_norm = np.linalg.norm(current_t)
    
    # Thresholds
    MIN_ROTATION = 0.05  # ~3 degrees
    MIN_TRANSLATION = 0.02  # relative to previous keyframe
    MIN_FEATURES = 50
    
    return (rotation_angle > MIN_ROTATION or 
            translation_norm > MIN_TRANSLATION or 
            num_features < MIN_FEATURES)
```

---

### 7. Limited Outlier Rejection
**Location:** `src/FeaturesFlow.py:12-29`, `src/VOEstimator.py:66-81`

**Current Approach:**
- Only forward-backward consistency check with threshold 1.0
- RANSAC in `findEssentialMat` (good)
- No additional validation

**Missing:**
- Parallax check (reject points too close to epipole)
- Reprojection error filtering after pose recovery
- Minimum inliers check

**Suggestion:**
```python
def validate_essential_matrix(self, E, mask, x0, x1, min_inliers=30):
    """Validate essential matrix estimation quality"""
    num_inliers = np.sum(mask)
    
    if num_inliers < min_inliers:
        return False, "Insufficient inliers"
    
    inlier_ratio = num_inliers / len(mask)
    if inlier_ratio < 0.3:
        return False, "Too many outliers"
    
    return True, "OK"

def check_parallax(self, x0, x1, min_parallax=1.0):
    """Check if points have sufficient parallax (degrees)"""
    angles = []
    for i in range(len(x0)):
        p0 = x0[i] / np.linalg.norm(x0[i])
        p1 = x1[i] / np.linalg.norm(x1[i])
        angle = np.arccos(np.clip(np.dot(p0, p1), -1, 1))
        angles.append(np.degrees(angle))
    
    median_parallax = np.median(angles)
    return median_parallax > min_parallax
```

---

### 8. Poor Feature Re-detection Strategy
**Location:** `src/VOEstimator.py:109-111`

```python
if tracked is None or len(tracked[1]) < 10:
    self.p0 = None
    return  # <-- Just skips the frame!
```

**Problem:** When tracking fails, the frame is completely skipped.

**Impact:** 
- Lost trajectory segments
- Gaps in pose estimation
- Cannot recover from tracking failure

**Better Approach:**
```python
if tracked is None or len(tracked[1]) < 10:
    # Re-detect features immediately in current frame
    self.p0 = detect_features(self.curr_frame)
    # Mark that we lost tracking (maybe increase uncertainty)
    self.tracking_lost = True
    return  # Skip pose update but maintain features

# Reset tracking status when sufficient features are available
if len(tracked[1]) >= 50 and self.tracking_lost:
    self.tracking_lost = False
```

---

## 🟢 Code Quality Issues

### 9. File Naming Inconsistency
**Location:** File system

**Problem:**
- Files: Both `Plotter.py` and `plotter.py` exist
- Import in code: `from src.Plotter import plot_trajectory_with_time_slider`
- Windows is case-insensitive, but Linux/Mac are not

**Impact:** Code will fail on Linux/Mac systems.

**Fix:** Standardize all Python modules to lowercase:
- `Plotter.py` → `plotter.py`
- Update all imports accordingly
- Python convention: modules are lowercase, classes are PascalCase

---

### 10. Commented Dead Code
**Location:** `main.py:49-54`

```python
# plot_trajectory_with_time_slider(
#     timestamps=gt_timestamps,
#     positions=gt_positions,
#     trajectory_label="Estimated Trajectory",
#     plot_3d=plot_3d
# )
```

**Problem:** Commented code clutters the file.

**Fix:** 
- Remove if not needed
- If needed for debugging, move to a separate debug/test file
- Document why it's kept if there's a specific reason

---

### 11. Unused Config Structure
**Location:** Workspace root

**Problem:**
- Workspace structure mentions `config.json` but file doesn't exist
- `ConfigManager` only loads calibration YAMLs, not algorithm parameters
- Algorithm parameters are scattered and hard-coded

**Suggestion:** Create unified configuration:

```json
{
    "features": {
        "max_corners": 1000,
        "quality_level": 0.1,
        "min_distance": 5,
        "block_size": 7
    },
    "tracking": {
        "fb_threshold": 1.0,
        "min_features": 10,
        "redetect_threshold": 50
    },
    "pose_estimation": {
        "ransac_prob": 0.999,
        "ransac_threshold": 0.01,
        "min_inliers": 30,
        "min_parallax_deg": 1.0
    },
    "keyframe": {
        "min_rotation_rad": 0.05,
        "min_translation": 0.02,
        "min_features_threshold": 50
    }
}
```

---

### 12. No Error Handling
**Location:** Throughout codebase

**Missing:**
- Try-except blocks for file I/O
- Validation of data shapes and ranges
- Handling of edge cases (empty dataframes, corrupted images)
- Graceful degradation

**Example Issues:**
```python
# DataManager.py - No validation
self.imu_df = pd.read_csv(os.path.join(dataset_path, "dvs-imu.csv"))

# VOEstimator.py - No null checks
self.camera_matrix = self.config_manager.get_camera_matrix()

# FeaturesFlow.py - No empty array checks
fb_err = np.linalg.norm(p0 - p0_est, axis=2).reshape(-1)
```

**Suggestions:**
```python
def load_imu_data(self, dataset_path):
    """Load IMU data with error handling"""
    imu_path = os.path.join(dataset_path, "dvs-imu.csv")
    try:
        if not os.path.exists(imu_path):
            self.logger.warning(f"IMU file not found: {imu_path}")
            return None
        df = pd.read_csv(imu_path)
        if df.empty:
            self.logger.warning("IMU data is empty")
            return None
        return df
    except Exception as e:
        self.logger.error(f"Error loading IMU data: {e}")
        return None
```

---

## 💡 Missing Features

### 13. No Bundle Adjustment
**Impact:** Accumulated errors grow unbounded without optimization.

**Suggestion:** Implement local bundle adjustment:
- Keep sliding window of last N keyframes
- Optimize camera poses and 3D point positions jointly
- Use tools like g2o, Ceres, or scipy.optimize

---

### 14. No Loop Closure
**Impact:** For longer sequences, drift becomes significant.

**Note:** May not be critical for short outdoor sequences, but important for longer trajectories.

**Suggestion:** 
- Implement bag-of-words for place recognition (DBoW2, DBoW3)
- Detect when camera returns to previously visited location
- Perform pose graph optimization to distribute accumulated error

---

### 15. No Runtime Visualization
**Current:** Only final trajectory plot after processing.

**Missing:** Real-time feedback during processing:
- Current frame with tracked features
- Feature distribution quality
- Number of inliers/outliers
- Trajectory plot updating in real-time

**Benefit:** Helps debug issues and tune parameters.

**Suggestion:**
```python
def visualize_tracking(self, frame, p0, p1, mask=None):
    """Draw tracked features on frame"""
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        # Green for inliers, red for outliers
        for i, (pt0, pt1) in enumerate(zip(p0, p1)):
            color = (0, 255, 0) if mask[i] else (0, 0, 255)
            cv2.circle(vis, tuple(pt1.astype(int)), 3, color, -1)
            cv2.line(vis, tuple(pt0.astype(int)), tuple(pt1.astype(int)), color, 1)
    cv2.imshow('Feature Tracking', vis)
    cv2.waitKey(1)
```

---

### 16. No Scale Drift Correction
**Problem:** Even with correct initial scale, monocular VO suffers from scale drift over time.

**Current:** Single scale factor (currently `/10`) applied uniformly.

**Suggestion:** 
- Periodically re-estimate scale from ground truth (if available)
- Use IMU accelerometer for continuous scale correction
- Track scale uncertainty and issue warnings when confidence is low

---

## 📋 Recommended Implementation Priority

### **Phase 1: Critical Fixes (Essential for Correctness)**
1. ✅ Fix rotation accumulation in trajectory calculation
2. ✅ Implement proper coordinate frame transformations  
3. ✅ Return and use rotation matrix from `estimate_pose()`
4. ✅ Remove arbitrary `/10` scale, implement GT-based scale alignment
5. ✅ Fix feature re-detection strategy (don't skip frames)

**Expected Impact:** Trajectory will be geometrically correct and comparable to ground truth.

---

### **Phase 2: Robustness Improvements**
6. ✅ Implement keyframe selection
7. ✅ Add parallax checking before pose estimation
8. ✅ Implement additional outlier rejection (reprojection error)
9. ✅ Add minimum tracked features threshold with warnings
10. ✅ Better validation of essential matrix quality

**Expected Impact:** More stable tracking, reduced drift, fewer failures.

---

### **Phase 3: Code Quality**
11. ✅ Fix file naming inconsistency (`Plotter.py` → `plotter.py`)
12. ✅ Create unified configuration file (`config.json`)
13. ✅ Remove commented dead code
14. ✅ Add comprehensive error handling
15. ✅ Add logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)

**Expected Impact:** More maintainable, portable, and debuggable code.

---

### **Phase 4: Advanced Features (After IMU Integration)**
16. ⏳ IMU integration for rotation and scale
17. ⏳ Implement local bundle adjustment
18. ⏳ Add runtime visualization
19. ⏳ Scale drift correction mechanism
20. ⏳ Loop closure detection (if needed for long sequences)

**Expected Impact:** Significantly reduced drift, metric scale, production-ready system.

---

## 📊 Current Pipeline Flow

```
DataManager (CSV Reading)
    ↓
VOEstimator.run()
    ↓
For each image frame:
    ↓
    FeaturesFlow.detect_features() [if needed]
    ↓
    FeaturesFlow.LK_optical_flow() [track features]
    ↓
    VOEstimator.estimate_pose() [Essential Matrix + recoverPose]
    ↓
    Store translation only (❌ rotation discarded)
    ↓
VOEstimator.calculate_trajectory()
    ↓
    Cumsum translations with arbitrary scale (❌ wrong reference frame)
    ↓
Plot results
```

---

## 🎯 Recommended Target Flow

```
DataManager (CSV Reading)
    ↓
VOEstimator.run()
    ↓
For each event:
    ↓
    [IMU] Update attitude estimate
    ↓
    [Image] Process frame:
        ↓
        Detect/Track features
        ↓
        [Keyframe decision] Skip if not keyframe
        ↓
        Check parallax and feature quality
        ↓
        Estimate Essential Matrix with RANSAC
        ↓
        Validate inliers count
        ↓
        Recover pose (R, t)
        ↓
        ✅ Store both R and t
        ↓
VOEstimator.calculate_trajectory()
    ↓
    Estimate scale from GT (first N frames)
    ↓
    For each keyframe:
        ↓
        Accumulate rotation: R_world = R_world @ R_rel
        ↓
        Transform translation: pos += R_world @ (scale * t)
    ↓
[Optional] Bundle adjustment every M keyframes
    ↓
Plot results with proper scale and orientation
```

---

## Notes

- This review excludes IMU integration as requested
- Many issues stem from the fundamental problem of not accumulating rotation
- Once rotation is properly handled, trajectory quality should improve dramatically
- Scale ambiguity is inherent to monocular VO and requires external reference (GT or IMU)

