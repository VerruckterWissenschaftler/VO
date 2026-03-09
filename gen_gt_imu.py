"""
gen_gt_imu.py
-------------
Generate a synthetic IMU CSV from ground-truth poses.

Acceleration  : double finite-diff of GT positions → world-frame → add Gaussian noise
Angular velocity : taken directly from the real dvs-imu.csv (interpolated to GT timestamps)

Output columns match dvs-imu.csv so it can be used as a drop-in replacement.

Usage
-----
    python gen_gt_imu.py                        # uses defaults below
    python gen_gt_imu.py --acc-noise 0.05       # custom noise std (m/s²)
    python gen_gt_imu.py --out my_imu.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

# ─────────────────────────── defaults ────────────────────────────────────────

DATASET_DIR  = "data/outdoor_forward_1_snapdragon_with_gt"
GT_CSV       = os.path.join(DATASET_DIR, "groundtruth-pose.csv")
IMU_CSV      = os.path.join(DATASET_DIR, "dvs-imu.csv")
DEFAULT_OUT  = os.path.join(DATASET_DIR, "gt-imu.csv")

# Noise parameters (Gaussian, applied independently per axis)
DEFAULT_ACC_NOISE_STD   = 2     # m/s²  — acceleration noise std
DEFAULT_GYRO_NOISE_STD  = 0.0   # rad/s — angular velocity noise std (0 = no extra noise)

# Bias parameters (constant offset added after noise, models sensor bias)
DEFAULT_ACC_BIAS        = [0.3, -0.2, 0.5]   # m/s²  — accel bias per axis [x, y, z]
DEFAULT_GYRO_BIAS       = [0.0, 0.0, 0.0]   # rad/s — gyro bias per axis [x, y, z]

RANDOM_SEED = 42


# ─────────────────────────── helpers ─────────────────────────────────────────

def finite_diff(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-order finite difference along axis 0. Returns (mid_times, derivatives)."""
    dt   = np.diff(times)
    dt   = np.where(dt < 1e-9, 1e-9, dt)
    dvdt = np.diff(values, axis=0) / dt[:, None]
    t_mid = times[1:]          # derivative lives at right edge of each interval
    return t_mid, dvdt


def interp_cols(target_t: np.ndarray, src_t: np.ndarray,
                src_vals: np.ndarray) -> np.ndarray:
    """Interpolate each column of src_vals onto target_t."""
    return np.column_stack([
        np.interp(target_t, src_t, src_vals[:, i])
        for i in range(src_vals.shape[1])
    ])


# ─────────────────────────── main ────────────────────────────────────────────

def generate(gt_csv: str, imu_csv: str, out_csv: str,
             acc_noise_std: float, gyro_noise_std: float, seed: int,
             acc_bias: list[float] | None = None,
             gyro_bias: list[float] | None = None) -> None:
    rng = np.random.default_rng(seed)
    acc_bias_vec  = np.array(acc_bias  if acc_bias  is not None else [0.0, 0.0, 0.0])
    gyro_bias_vec = np.array(gyro_bias if gyro_bias is not None else [0.0, 0.0, 0.0])

    # ── Load GT ──────────────────────────────────────────────────────────────
    gt  = pd.read_csv(gt_csv)
    gt_t = gt["Time"].values.astype(np.float64)
    gt_p = gt[["pose.position.x", "pose.position.y", "pose.position.z"]].values

    # ── Derive GT acceleration (double finite-diff of position) ───────────────
    t_vel, gt_vel = finite_diff(gt_t, gt_p)          # velocity at t_vel
    t_acc, gt_acc = finite_diff(t_vel, gt_vel)        # acceleration at t_acc

    # ── Load real IMU for angular velocity ────────────────────────────────────
    imu = pd.read_csv(imu_csv)
    imu_t   = imu["Time"].values.astype(np.float64)
    imu_omega = imu[["angular_velocity.x",
                      "angular_velocity.y",
                      "angular_velocity.z"]].values

    # ── Build output at GT acceleration timestamps ────────────────────────────
    # Interpolate real gyro onto GT-acc timestamps
    omega_interp = interp_cols(t_acc, imu_t, imu_omega)

    # Add noise + bias (bias is a constant offset, noise is zero-mean random)
    acc_noisy   = gt_acc       + rng.normal(0.0, acc_noise_std,  gt_acc.shape)       + acc_bias_vec
    omega_noisy = omega_interp + rng.normal(0.0, gyro_noise_std, omega_interp.shape) + gyro_bias_vec

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    n = len(t_acc)
    zeros = np.zeros(n)
    ones  = np.ones(n)

    out = pd.DataFrame({
        "Time":                         t_acc,
        "header.seq":                   np.arange(n),
        "header.stamp.secs":            t_acc.astype(int),
        "header.stamp.nsecs":           ((t_acc % 1) * 1e9).astype(int),
        "header.frame_id":              ["snappy_imu"] * n,
        "orientation.x":                zeros,
        "orientation.y":                zeros,
        "orientation.z":                zeros,
        "orientation.w":                zeros,
        **{f"orientation_covariance_{i}": zeros for i in range(9)},
        "angular_velocity.x":           omega_noisy[:, 0],
        "angular_velocity.y":           omega_noisy[:, 1],
        "angular_velocity.z":           omega_noisy[:, 2],
        **{f"angular_velocity_covariance_{i}": zeros for i in range(9)},
        "linear_acceleration.x":        acc_noisy[:, 0],
        "linear_acceleration.y":        acc_noisy[:, 1],
        "linear_acceleration.z":        acc_noisy[:, 2],
        **{f"linear_acceleration_covariance_{i}": zeros for i in range(9)},
    })

    out.to_csv(out_csv, index=False)
    print(f"Written {n} rows → {out_csv}")
    print(f"  GT acc range  x=[{gt_acc[:,0].min():.3f}, {gt_acc[:,0].max():.3f}]"
          f"  y=[{gt_acc[:,1].min():.3f}, {gt_acc[:,1].max():.3f}]"
          f"  z=[{gt_acc[:,2].min():.3f}, {gt_acc[:,2].max():.3f}]  m/s²")
    print(f"  Noise std  acc={acc_noise_std} m/s²  gyro={gyro_noise_std} rad/s")
    print(f"  Bias       acc={acc_bias_vec.tolist()} m/s²  gyro={gyro_bias_vec.tolist()} rad/s")


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GT-derived synthetic IMU CSV")
    parser.add_argument("--gt",          default=GT_CSV,
                        help="Ground-truth pose CSV")
    parser.add_argument("--imu",         default=IMU_CSV,
                        help="Real IMU CSV (for angular velocity)")
    parser.add_argument("--out",         default=DEFAULT_OUT,
                        help="Output CSV path")
    parser.add_argument("--acc-noise",   type=float, default=DEFAULT_ACC_NOISE_STD,
                        help="Acceleration noise std (m/s², default %(default)s)")
    parser.add_argument("--gyro-noise",  type=float, default=DEFAULT_GYRO_NOISE_STD,
                        help="Angular velocity noise std (rad/s, default %(default)s)")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help="Random seed (default %(default)s)")
    parser.add_argument("--acc-bias",    type=float, nargs=3, default=DEFAULT_ACC_BIAS,
                        metavar=("BX", "BY", "BZ"),
                        help="Accel bias [x y z] in m/s² (default %(default)s)")
    parser.add_argument("--gyro-bias",   type=float, nargs=3, default=DEFAULT_GYRO_BIAS,
                        metavar=("BX", "BY", "BZ"),
                        help="Gyro bias [x y z] in rad/s (default %(default)s)")
    args = parser.parse_args()

    generate(
        gt_csv=args.gt,
        imu_csv=args.imu,
        out_csv=args.out,
        acc_noise_std=args.acc_noise,
        gyro_noise_std=args.gyro_noise,
        seed=args.seed,
        acc_bias=args.acc_bias,
        gyro_bias=args.gyro_bias,
    )
