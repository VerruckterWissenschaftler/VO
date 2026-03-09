"""
extract_bag.py — Unpack a ROS1 .bag file into the dataset folder format
used by this VO pipeline.

Usage:
    python extract_bag.py <path/to/file.bag> [options]

Options:
    --image-topic   IMAGE_TOPIC   Image topic to extract (default: auto-detect)
    --imu-topic     IMU_TOPIC     IMU topic (default: auto-detect)
    --gt-topic      GT_TOPIC      Ground-truth topic (default: auto-detect)
    --out-dir       DIR           Output directory (default: bag filename without .bag)
    --duration      SECS          Limit extraction to first N seconds (default: full bag)

Output folder structure (mirrors outdoor_forward_1_davis_with_gt):
    <out_dir>/
        dvs-image_raw.csv
        dvs-imu.csv
        groundtruth-pose.csv
        params.yml
"""

import argparse
import os
import sys
import csv
import numpy as np
from pathlib import Path

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("ERROR: rosbags not installed. Run: pip install rosbags")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Topic auto-detection heuristics
# ---------------------------------------------------------------------------

_IMAGE_KEYWORDS  = ["image", "cam", "camera", "stereo_l", "left", "img"]
_IMU_KEYWORDS    = ["imu", "inertial"]
_GT_KEYWORDS     = ["groundtruth", "ground_truth", "optitrack", "vicon",
                    "mocap", "pose", "odometry", "gt"]
_IMAGE_MSGTYPES  = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
_IMU_MSGTYPES    = {"sensor_msgs/msg/Imu"}
_GT_MSGTYPES     = {"geometry_msgs/msg/PoseStamped",
                    "nav_msgs/msg/Odometry",
                    "geometry_msgs/msg/PoseWithCovarianceStamped"}


def _score(topic: str, keywords: list[str]) -> int:
    topic_lower = topic.lower()
    return sum(k in topic_lower for k in keywords)


def auto_select(connections, msgtype_set: set, keywords: list[str],
                label: str) -> str | None:
    candidates = [c for c in connections if c.msgtype in msgtype_set]
    if not candidates:
        return None
    scored = sorted(candidates, key=lambda c: _score(c.topic, keywords), reverse=True)
    chosen = scored[0].topic
    if len(candidates) > 1:
        print(f"  [{label}] Multiple candidates: "
              f"{[c.topic for c in candidates]} -> chose '{chosen}'")
    return chosen


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# ---------------------------------------------------------------------------
# Image encoding helper (matches DavisCsvReader expectation)
# ---------------------------------------------------------------------------

def _encode_image_data(pixel_array: np.ndarray) -> str:
    """Return Python bytes-repr string that DavisCsvReader can ast.literal_eval."""
    return repr(pixel_array.tobytes())


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(bag_path: str, image_topic: str | None, imu_topic: str | None,
            gt_topic: str | None, out_dir: str | None, duration: float | None):

    bag_path = Path(bag_path).resolve()
    if not bag_path.exists():
        print(f"ERROR: bag file not found: {bag_path}")
        sys.exit(1)

    # Output directory defaults to bag stem next to the bag file
    if out_dir is None:
        out_dir = bag_path.parent / bag_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(str(bag_path)) as bag:

        # ----------------------------------------------------------------
        # Enumerate topics
        # ----------------------------------------------------------------
        print("\nTopics in bag:")
        for conn in bag.connections:
            print(f"  {conn.topic}  ({conn.msgtype})")

        # Auto-detect topics
        if image_topic is None:
            image_topic = auto_select(bag.connections, _IMAGE_MSGTYPES,
                                      _IMAGE_KEYWORDS, "image")
        if imu_topic is None:
            imu_topic = auto_select(bag.connections, _IMU_MSGTYPES,
                                    _IMU_KEYWORDS, "imu")
        if gt_topic is None:
            gt_topic = auto_select(bag.connections, _GT_MSGTYPES,
                                   _GT_KEYWORDS, "gt")

        print(f"\nSelected topics:")
        print(f"  image : {image_topic}")
        print(f"  imu   : {imu_topic}")
        print(f"  gt    : {gt_topic}")

        wanted = {t for t in [image_topic, imu_topic, gt_topic] if t}
        conns = [c for c in bag.connections if c.topic in wanted]

        # ----------------------------------------------------------------
        # Collect all messages (two-pass: first to find time range)
        # ----------------------------------------------------------------
        first_ts: float | None = None
        last_ts:  float | None = None

        for conn, ts_ns, raw in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            t = _stamp_to_sec(msg.header.stamp)
            if first_ts is None or t < first_ts:
                first_ts = t
            if last_ts is None or t > last_ts:
                last_ts = t

        if first_ts is None:
            print("ERROR: no messages found in selected topics")
            sys.exit(1)

        end_ts = (first_ts + duration) if duration is not None else last_ts
        total  = last_ts - first_ts
        print(f"\nBag time range : {first_ts:.3f} -> {last_ts:.3f}  "
              f"({total:.1f} s)")
        print(f"Extracting     : {first_ts:.3f} -> {end_ts:.3f}  "
              f"({min(total, end_ts - first_ts):.1f} s)")

        # ----------------------------------------------------------------
        # Write CSVs
        # ----------------------------------------------------------------
        img_path = out_dir / "dvs-image_raw.csv"
        imu_path = out_dir / "dvs-imu.csv"
        gt_path  = out_dir / "groundtruth-pose.csv"

        img_header = ["Time", "header.seq", "header.stamp.secs",
                      "header.stamp.nsecs", "header.frame_id",
                      "height", "width", "encoding", "is_bigendian", "step", "data"]
        imu_header = ["Time", "header.seq", "header.stamp.secs",
                      "header.stamp.nsecs", "header.frame_id",
                      "orientation.x", "orientation.y", "orientation.z", "orientation.w",
                      "orientation_covariance_0", "orientation_covariance_1",
                      "orientation_covariance_2", "orientation_covariance_3",
                      "orientation_covariance_4", "orientation_covariance_5",
                      "orientation_covariance_6", "orientation_covariance_7",
                      "orientation_covariance_8",
                      "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                      "angular_velocity_covariance_0", "angular_velocity_covariance_1",
                      "angular_velocity_covariance_2", "angular_velocity_covariance_3",
                      "angular_velocity_covariance_4", "angular_velocity_covariance_5",
                      "angular_velocity_covariance_6", "angular_velocity_covariance_7",
                      "angular_velocity_covariance_8",
                      "linear_acceleration.x", "linear_acceleration.y",
                      "linear_acceleration.z",
                      "linear_acceleration_covariance_0", "linear_acceleration_covariance_1",
                      "linear_acceleration_covariance_2", "linear_acceleration_covariance_3",
                      "linear_acceleration_covariance_4", "linear_acceleration_covariance_5",
                      "linear_acceleration_covariance_6", "linear_acceleration_covariance_7",
                      "linear_acceleration_covariance_8"]
        gt_header  = ["Time", "header.seq", "header.stamp.secs",
                      "header.stamp.nsecs", "header.frame_id",
                      "pose.position.x", "pose.position.y", "pose.position.z",
                      "pose.orientation.x", "pose.orientation.y",
                      "pose.orientation.z", "pose.orientation.w"]

        n_img = n_imu = n_gt = 0

        with (open(img_path, "w", newline="", encoding="utf-8") as f_img,
              open(imu_path, "w", newline="", encoding="utf-8") as f_imu,
              open(gt_path,  "w", newline="", encoding="utf-8") as f_gt):

            w_img = csv.writer(f_img)
            w_imu = csv.writer(f_imu)
            w_gt  = csv.writer(f_gt)
            w_img.writerow(img_header)
            w_imu.writerow(imu_header)
            w_gt.writerow(gt_header)

            for conn, ts_ns, raw in bag.messages(connections=conns):
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                hdr = msg.header
                t   = _stamp_to_sec(hdr.stamp)

                if t < first_ts or t > end_ts:
                    continue

                # ---- IMAGE ----
                if conn.topic == image_topic:
                    pixels = np.asarray(msg.data, dtype=np.uint8)
                    # Compressed images not supported here — bag uses raw Image
                    w_img.writerow([
                        t,
                        hdr.seq,
                        hdr.stamp.sec,
                        hdr.stamp.nanosec,
                        hdr.frame_id,
                        msg.height,
                        msg.width,
                        msg.encoding,
                        int(msg.is_bigendian),
                        msg.step,
                        _encode_image_data(pixels),
                    ])
                    n_img += 1

                # ---- IMU ----
                elif conn.topic == imu_topic:
                    o  = msg.orientation
                    av = msg.angular_velocity
                    la = msg.linear_acceleration
                    oc = list(msg.orientation_covariance)
                    vc = list(msg.angular_velocity_covariance)
                    ac = list(msg.linear_acceleration_covariance)
                    w_imu.writerow([
                        t,
                        hdr.seq, hdr.stamp.sec, hdr.stamp.nanosec, hdr.frame_id,
                        o.x,  o.y,  o.z,  o.w,
                        *oc,
                        av.x, av.y, av.z,
                        *vc,
                        la.x, la.y, la.z,
                        *ac,
                    ])
                    n_imu += 1

                # ---- GROUND TRUTH ----
                elif conn.topic == gt_topic:
                    # Unwrap pose regardless of message type:
                    #   PoseStamped              → msg.pose          (has .position)
                    #   Odometry                 → msg.pose.pose     (PoseWithCovariance → Pose)
                    #   PoseWithCovarianceStamped→ msg.pose.pose
                    inner = msg.pose
                    if hasattr(inner, "pose"):          # PoseWithCovariance wrapper
                        pose = inner.pose
                    elif hasattr(inner, "position"):    # already a Pose
                        pose = inner
                    else:
                        pose = inner
                    pos = pose.position
                    ori = pose.orientation
                    w_gt.writerow([
                        t,
                        hdr.seq, hdr.stamp.sec, hdr.stamp.nanosec, hdr.frame_id,
                        pos.x, pos.y, pos.z,
                        ori.x, ori.y, ori.z, ori.w,
                    ])
                    n_gt += 1

        print(f"\nWrote {n_img} images  -> {img_path.name}")
        print(f"Wrote {n_imu} IMU rows -> {imu_path.name}")
        print(f"Wrote {n_gt}  GT rows  -> {gt_path.name}")

        # ----------------------------------------------------------------
        # Auto-generate params.yml
        # ----------------------------------------------------------------
        _write_params(out_dir, first_ts, total)


def _write_params(out_dir: Path, start_time: float, total_duration: float):
    params_path = out_dir / "params.yml"
    if params_path.exists():
        print(f"\nparams.yml already exists — skipping (delete to regenerate)")
        return

    content = f"""\
use_imu: true          # Enable IMU data processing
use_vo: true           # Enable visual odometry
use_ukf: false         # Enable UKF filtering
debug_ukf: false       # Attach UKFDebugger after run
use_gt_imu: false      # Replace gyro orientation with GT orientation (testing only)
start_time: {start_time:.5f}
show_frames: false     # Display OpenCV window during processing
duration: 10.0         # Seconds to process (bag total: {total_duration:.1f} s)
matcher: sift          # sift | orb | lk | lightglue | loftr
init_height: 0.01      # Initial height in metres; null = use GT Z

UKF:
  acc_proc_cov: 0.01
  gyro_proc_cov: 0.0025
  meas_noise: 0.5
  init_uncertainty: 1.0e-3
  vel_decay_rate: 2.0
"""
    params_path.write_text(content, encoding="utf-8")
    print(f"\nCreated {params_path.name}")
    print(f"  start_time : {start_time:.5f}")
    print(f"  duration   : 10.0  (bag total: {total_duration:.1f} s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unpack ROS1 .bag -> VO pipeline dataset folder"
    )
    parser.add_argument("bag", help="Path to .bag file")
    parser.add_argument("--image-topic", default=None)
    parser.add_argument("--imu-topic",   default=None)
    parser.add_argument("--gt-topic",    default=None)
    parser.add_argument("--out-dir",     default=None,
                        help="Output folder (default: <bag stem> next to bag)")
    parser.add_argument("--duration",    type=float, default=None,
                        help="Extract only first N seconds")
    args = parser.parse_args()

    extract(
        bag_path     = args.bag,
        image_topic  = args.image_topic,
        imu_topic    = args.imu_topic,
        gt_topic     = args.gt_topic,
        out_dir      = args.out_dir,
        duration     = args.duration,
    )
