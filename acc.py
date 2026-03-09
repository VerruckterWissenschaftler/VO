"""
IMU Acceleration Data Analysis with Advanced Filtering and Velocity Integration
================================================================================

This script analyzes acceleration data from both UKF world-frame and raw IMU sensors.
It implements multiple filtering techniques commonly used in flight controllers:

FILTERING:
1. Rolling Mean (Boxcar) Filter - Simple but has frequency leakage
2. Median Filter - Non-linear filter excellent at removing outliers and spike noise
   - Preserves edges better than linear filters
   - Ideal for impulsive disturbances and salt-and-pepper noise
3. Static Notch Filter - Removes specific frequency bands (e.g., motor idle at 150 Hz)
4. Dynamic Notch Filter - FFT-based tracking of dominant frequencies
   - Automatically tracks motor noise as throttle changes
   - Used in modern flight controllers like Betaflight

VELOCITY INTEGRATION:
- Integrates filtered acceleration to velocity using trapezoidal rule
- Applies drift correction (linear detrending) to reduce integration errors
- WARNING: Integration amplifies low-frequency noise and bias errors!

Reference: Flight controller filtering techniques from Betaflight/ArduPilot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

# ========================= CONFIGURATION =========================
# Filter Parameters (adjust these for your data)
STATIC_NOTCH_FREQ = 150      # Hz (typical motor idle frequency)
NOTCH_Q_FACTOR = 30          # Quality factor (higher = narrower notch)
ROLLING_WINDOW = 20          # Samples for rolling mean
MEDIAN_KERNEL = 51            # Median filter kernel size (must be odd)
DYNAMIC_NOTCH_RANGE = (50, 300)  # Hz range to search for motor noise

# Plot Time Interval (None = plot all data, or specify (start, end) in seconds)
PLOT_TIME_INTERVAL = (542, 555)    # e.g., (542, 545) to plot only 542-545 seconds
                             # or None to plot entire dataset
# =================================================================

# Read the UKF acceleration log
df_ukf = pd.read_csv('data/outdoor_forward_1_snapdragon_with_gt/ukf_accel_log.csv')

# Read the raw IMU data
df_imu = pd.read_csv('data/outdoor_forward_1_snapdragon_with_gt/dvs-imu.csv')

# ===== FILTERING FUNCTIONS =====
def get_sampling_rate(time_series):
    """Calculate sampling rate from time series"""
    dt = np.diff(time_series).mean()
    return 1.0 / dt

def apply_notch_filter(data, notch_freq, fs, Q=30):
    """
    Apply static notch filter to remove specific frequency
    Args:
        data: Input signal
        notch_freq: Frequency to notch out (Hz)
        fs: Sampling rate (Hz)
        Q: Quality factor (higher = narrower notch, default=30)
    """
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, data)

def apply_median_filter(data, kernel_size=5):
    """
    Apply median filter to remove outliers and impulsive noise
    Args:
        data: Input signal
        kernel_size: Size of the median filter kernel (must be odd)
    Returns:
        Filtered signal
    Note:
        Median filters are excellent at removing:
        - Outliers and spike noise
        - Salt-and-pepper noise
        - Impulsive disturbances
        While preserving edges better than linear filters
    """
    return signal.medfilt(data, kernel_size=kernel_size)

def find_dominant_frequency(data, fs, freq_range=(50, 300)):
    """
    Find dominant frequency in the signal using FFT (for dynamic notch)
    Args:
        data: Input signal
        fs: Sampling rate (Hz)
        freq_range: Tuple of (min_freq, max_freq) to search
    """
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1/fs)
    
    # Only look at positive frequencies
    pos_mask = xf > 0
    xf_pos = xf[pos_mask]
    yf_pos = np.abs(yf[pos_mask])
    
    # Limit to frequency range
    range_mask = (xf_pos >= freq_range[0]) & (xf_pos <= freq_range[1])
    xf_range = xf_pos[range_mask]
    yf_range = yf_pos[range_mask]
    
    if len(yf_range) > 0:
        dominant_idx = np.argmax(yf_range)
        return xf_range[dominant_idx]
    return None

def apply_dynamic_notch_filter(data, fs, window_size=1000, freq_range=(50, 300), Q=30):
    """
    Apply dynamic notch filter that tracks dominant frequency
    Simulates FFT-based dynamic notch used in flight controllers
    """
    filtered = np.copy(data)
    
    # Process in overlapping windows
    step = window_size // 2
    for i in range(0, len(data) - window_size, step):
        window_data = data[i:i+window_size]
        
        # Find dominant frequency in this window
        dominant_freq = find_dominant_frequency(window_data, fs, freq_range)
        
        if dominant_freq:
            # Apply notch at the dominant frequency
            filtered[i:i+window_size] = apply_notch_filter(
                filtered[i:i+window_size], dominant_freq, fs, Q
            )
    
    return filtered

def integrate_acceleration_to_velocity(acceleration, time, initial_velocity=0.0):
    """
    Integrate acceleration to obtain velocity using cumulative trapezoidal integration
    Args:
        acceleration: Acceleration array (m/s^2)
        time: Time array (seconds)
        initial_velocity: Initial velocity (m/s), default is 0.0
    Returns:
        Velocity array (m/s)
    Note:
        Uses trapezoidal rule for numerical integration:
        v(t) = v0 + ∫a(t)dt
        
        Warning: Integration amplifies low-frequency drift in acceleration data.
        For accurate velocity estimation, ensure acceleration is properly filtered
        and bias-corrected. High-pass filtering may be needed to remove DC bias.
    """
    # Compute time differences
    dt = np.diff(time)
    
    # Trapezoidal integration: v[i] = v[i-1] + (a[i] + a[i-1])/2 * dt[i]
    velocity = np.zeros_like(acceleration)
    velocity[0] = initial_velocity
    
    for i in range(1, len(acceleration)):
        velocity[i] = velocity[i-1] + 0.5 * (acceleration[i] + acceleration[i-1]) * dt[i-1]
    
    return velocity

def remove_velocity_drift(velocity, time, method='linear'):
    """
    Remove linear drift from integrated velocity
    Args:
        velocity: Velocity array with drift
        time: Time array
        method: 'linear' (detrend), 'highpass' (remove DC), or 'none'
    Returns:
        Drift-corrected velocity
    Note:
        Integration of noisy acceleration introduces drift. This function
        attempts to remove systematic drift trends.
    """
    if method == 'linear':
        # Remove linear trend (assumes zero mean velocity over long term)
        return signal.detrend(velocity, type='linear')
    elif method == 'highpass':
        # High-pass filter to remove DC component
        # (keeps oscillations, removes constant drift)
        fs = 1.0 / np.diff(time).mean()
        sos = signal.butter(2, 0.5, 'highpass', fs=fs, output='sos')
        return signal.sosfiltfilt(sos, velocity)
    else:
        return velocity

# ===== APPLY FILTERS TO IMU DATA =====
# Calculate sampling rate
fs_imu = get_sampling_rate(df_imu['Time'].values)
print(f"\nIMU Sampling Rate: {fs_imu:.2f} Hz")

# Apply median filters (removes outliers and spike noise)
print(f"Applying median filter (kernel size={MEDIAN_KERNEL})...")
df_imu['lax_median'] = apply_median_filter(df_imu['linear_acceleration.x'].values, MEDIAN_KERNEL)
df_imu['lay_median'] = apply_median_filter(df_imu['linear_acceleration.y'].values, MEDIAN_KERNEL)
df_imu['laz_median'] = apply_median_filter(df_imu['linear_acceleration.z'].values, MEDIAN_KERNEL)

# Apply static notch filters on median filtered data
print(f"Applying static notch filter at {STATIC_NOTCH_FREQ} Hz...")
df_imu['lax_notch'] = apply_notch_filter(df_imu['lax_median'].values, STATIC_NOTCH_FREQ, fs_imu, NOTCH_Q_FACTOR)
df_imu['lay_notch'] = apply_notch_filter(df_imu['lay_median'].values, STATIC_NOTCH_FREQ, fs_imu, NOTCH_Q_FACTOR)
df_imu['laz_notch'] = apply_notch_filter(df_imu['laz_median'].values, STATIC_NOTCH_FREQ, fs_imu, NOTCH_Q_FACTOR)

# Apply dynamic notch filters (tracks dominant frequency)
print("Applying dynamic notch filters (this may take a moment)...")
df_imu['lax_dynamic'] = apply_dynamic_notch_filter(df_imu['lax_median'].values, fs_imu, freq_range=DYNAMIC_NOTCH_RANGE, Q=NOTCH_Q_FACTOR)
df_imu['lay_dynamic'] = apply_dynamic_notch_filter(df_imu['lay_median'].values, fs_imu, freq_range=DYNAMIC_NOTCH_RANGE, Q=NOTCH_Q_FACTOR)
df_imu['laz_dynamic'] = apply_dynamic_notch_filter(df_imu['laz_median'].values, fs_imu, freq_range=DYNAMIC_NOTCH_RANGE, Q=NOTCH_Q_FACTOR)
print("Dynamic notch filtering complete!")

# ===== INTEGRATE ACCELERATIONS TO VELOCITIES =====
print("\nIntegrating filtered accelerations to velocities...")
time_imu = df_imu['Time'].values

# Integrate each filtered acceleration to get velocity
# Raw data
df_imu['vx_raw'] = integrate_acceleration_to_velocity(df_imu['linear_acceleration.x'].values, time_imu)
df_imu['vy_raw'] = integrate_acceleration_to_velocity(df_imu['linear_acceleration.y'].values, time_imu)
df_imu['vz_raw'] = integrate_acceleration_to_velocity(df_imu['linear_acceleration.z'].values, time_imu)

# Median filtered (best for velocity estimation)
df_imu['vx_median'] = integrate_acceleration_to_velocity(df_imu['lax_median'].values, time_imu)
df_imu['vy_median'] = integrate_acceleration_to_velocity(df_imu['lay_median'].values, time_imu)
df_imu['vz_median'] = integrate_acceleration_to_velocity(df_imu['laz_median'].values, time_imu)

# Detrend velocities to remove drift (linear detrending)
df_imu['vx_median_detrend'] = remove_velocity_drift(df_imu['vx_median'].values, time_imu, 'linear')
df_imu['vy_median_detrend'] = remove_velocity_drift(df_imu['vy_median'].values, time_imu, 'linear')
df_imu['vz_median_detrend'] = remove_velocity_drift(df_imu['vz_median'].values, time_imu, 'linear')

print("Velocity integration complete!")

# Rolling window size (adjust as needed)
window_size = ROLLING_WINDOW

# Calculate rolling averages for UKF data
df_ukf['ax_smooth'] = df_ukf['ax_world'].rolling(window=window_size, center=True).mean()
df_ukf['ay_smooth'] = df_ukf['ay_world'].rolling(window=window_size, center=True).mean()
df_ukf['az_smooth'] = df_ukf['az_world'].rolling(window=window_size, center=True).mean()

# Calculate rolling averages for IMU data
df_imu['lax_smooth'] = df_imu['linear_acceleration.x'].rolling(window=window_size, center=True).mean()
df_imu['lay_smooth'] = df_imu['linear_acceleration.y'].rolling(window=window_size, center=True).mean()
df_imu['laz_smooth'] = df_imu['linear_acceleration.z'].rolling(window=window_size, center=True).mean()

# Apply time interval filtering if specified
if PLOT_TIME_INTERVAL is not None:
    t_start, t_end = PLOT_TIME_INTERVAL
    df_ukf_plot = df_ukf[(df_ukf['t_s'] >= t_start) & (df_ukf['t_s'] <= t_end)].copy()
    df_imu_plot = df_imu[(df_imu['Time'] >= t_start) & (df_imu['Time'] <= t_end)].copy()
    interval_label = f" [{t_start}-{t_end}s]"
else:
    df_ukf_plot = df_ukf.copy()
    df_imu_plot = df_imu.copy()
    interval_label = ""

# Create figure with subplots - 2 columns for UKF and IMU
fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
fig.suptitle(f'Acceleration Analysis: UKF vs IMU with Advanced Filtering (Median, Notch Filters){interval_label}', 
             fontsize=15, fontweight='bold')

# ===== UKF PLOTS (Left Column) =====
# Plot UKF X acceleration
axes[0, 0].plot(df_ukf_plot['t_s'], df_ukf_plot['ax_world'], alpha=0.3, linewidth=0.5, label='Raw', color='blue')
axes[0, 0].plot(df_ukf_plot['t_s'], df_ukf_plot['ax_smooth'], linewidth=2, label=f'Rolling Mean (w={ROLLING_WINDOW})', color='darkblue')
axes[0, 0].set_ylabel('X Acceleration (m/s²)', fontsize=11)
axes[0, 0].set_ylim([-2, 10])
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('UKF - X-axis (ax_world)', loc='left', fontsize=10, fontweight='bold')

# Plot UKF Y acceleration
axes[1, 0].plot(df_ukf_plot['t_s'], df_ukf_plot['ay_world'], alpha=0.3, linewidth=0.5, label='Raw', color='green')
axes[1, 0].plot(df_ukf_plot['t_s'], df_ukf_plot['ay_smooth'], linewidth=2, label=f'Rolling Mean (w={ROLLING_WINDOW})', color='darkgreen')
axes[1, 0].set_ylabel('Y Acceleration (m/s²)', fontsize=11)
axes[1, 0].set_ylim([-2, 10])
axes[1, 0].legend(loc='upper right')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('UKF - Y-axis (ay_world)', loc='left', fontsize=10, fontweight='bold')

# Plot UKF Z acceleration
axes[2, 0].plot(df_ukf_plot['t_s'], df_ukf_plot['az_world'], alpha=0.3, linewidth=0.5, label='Raw', color='red')
axes[2, 0].plot(df_ukf_plot['t_s'], df_ukf_plot['az_smooth'], linewidth=2, label=f'Rolling Mean (w={ROLLING_WINDOW})', color='darkred')
axes[2, 0].set_ylabel('Z Acceleration (m/s²)', fontsize=11)
axes[2, 0].set_ylim([-2, 10])
axes[2, 0].set_xlabel('Time (s)', fontsize=12)
axes[2, 0].legend(loc='upper right')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_title('UKF - Z-axis (az_world)', loc='left', fontsize=10, fontweight='bold')

# ===== IMU PLOTS (Right Column) =====
# Plot IMU X acceleration with multiple filters
axes[0, 1].plot(df_imu_plot['Time'], df_imu_plot['linear_acceleration.x'], alpha=0.2, linewidth=0.5, label='Raw', color='lightblue')
# axes[0, 1].plot(df_imu_plot['Time'], df_imu_plot['lax_smooth'], linewidth=1.3, label=f'Rolling Mean (w={ROLLING_WINDOW})', color='blue', alpha=0.5)
axes[0, 1].plot(df_imu_plot['Time'], df_imu_plot['lax_median'], linewidth=1.5, label=f'Median (k={MEDIAN_KERNEL})', color='cyan', alpha=0.8)
# axes[0, 1].plot(df_imu_plot['Time'], df_imu_plot['lax_butter'], linewidth=2, label=f'Butterworth (fc={BUTTERWORTH_CUTOFF}Hz)', color='darkblue')
# axes[0, 1].plot(df_imu_plot['Time'], df_imu_plot['lax_notch'], linewidth=1.5, label=f'Static Notch ({STATIC_NOTCH_FREQ}Hz)', color='purple', alpha=0.7)
# axes[0, 1].plot(df_imu_plot['Time'], df_imu_plot['lax_dynamic'], linewidth=2, label='Dynamic Notch', color='red')
axes[0, 1].set_ylabel('X Acceleration (m/s²)', fontsize=11)
axes[0, 1].set_ylim([-2, 10])
axes[0, 1].legend(loc='upper right', fontsize=7)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('IMU - X-axis (linear_acceleration.x)', loc='left', fontsize=10, fontweight='bold')

# Plot IMU Y acceleration with multiple filters
axes[1, 1].plot(df_imu_plot['Time'], df_imu_plot['linear_acceleration.y'], alpha=0.2, linewidth=0.5, label='Raw', color='lightgreen')
# axes[1, 1].plot(df_imu_plot['Time'], df_imu_plot['lay_smooth'], linewidth=1.3, label=f'Rolling Mean (w={ROLLING_WINDOW})', color='green', alpha=0.5)
axes[1, 1].plot(df_imu_plot['Time'], df_imu_plot['lay_median'], linewidth=1.5, label=f'Median (k={MEDIAN_KERNEL})', color='cyan', alpha=0.8)
# axes[1, 1].plot(df_imu_plot['Time'], df_imu_plot['lay_butter'], linewidth=2, label=f'Butterworth (fc={BUTTERWORTH_CUTOFF}Hz)', color='darkgreen')
# axes[1, 1].plot(df_imu_plot['Time'], df_imu_plot['lay_notch'], linewidth=1.5, label=f'Static Notch ({STATIC_NOTCH_FREQ}Hz)', color='purple', alpha=0.7)
# axes[1, 1].plot(df_imu_plot['Time'], df_imu_plot['lay_dynamic'], linewidth=2, label='Dynamic Notch', color='red')
axes[1, 1].set_ylabel('Y Acceleration (m/s²)', fontsize=11)
axes[1, 1].set_ylim([-2, 10])
axes[1, 1].legend(loc='upper right', fontsize=7)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('IMU - Y-axis (linear_acceleration.y)', loc='left', fontsize=10, fontweight='bold')

# Plot IMU Z acceleration with multiple filters
axes[2, 1].plot(df_imu_plot['Time'], df_imu_plot['linear_acceleration.z'], alpha=0.2, linewidth=0.5, label='Raw', color='lightcoral')
# axes[2, 1].plot(df_imu_plot['Time'], df_imu_plot['laz_smooth'], linewidth=1.3, label=f'Rolling Mean (w={ROLLING_WINDOW})', color='red', alpha=0.5)
axes[2, 1].plot(df_imu_plot['Time'], df_imu_plot['laz_median'], linewidth=1.5, label=f'Median (k={MEDIAN_KERNEL})', color='cyan', alpha=0.8)
# axes[2, 1].plot(df_imu_plot['Time'], df_imu_plot['laz_butter'], linewidth=2, label=f'Butterworth (fc={BUTTERWORTH_CUTOFF}Hz)', color='darkred')
# axes[2, 1].plot(df_imu_plot['Time'], df_imu_plot['laz_notch'], linewidth=1.5, label=f'Static Notch ({STATIC_NOTCH_FREQ}Hz)', color='purple', alpha=0.7)
# axes[2, 1].plot(df_imu_plot['Time'], df_imu_plot['laz_dynamic'], linewidth=2, label='Dynamic Notch', color='red')
axes[2, 1].set_ylabel('Z Acceleration (m/s²)', fontsize=11)
axes[2, 1].set_ylim([-2, 10])
axes[2, 1].set_xlabel('Time (s)', fontsize=12)
axes[2, 1].legend(loc='upper right', fontsize=7)
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_title('IMU - Z-axis (linear_acceleration.z)', loc='left', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ===== CREATE VELOCITY PLOTS =====
fig_vel, axes_vel = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig_vel.suptitle(f'Integrated Velocity from Filtered IMU Accelerations{interval_label}', fontsize=16, fontweight='bold')

# Plot X velocity
axes_vel[0].plot(df_imu_plot['Time'], df_imu_plot['vx_raw'], alpha=0.3, linewidth=0.8, label='Raw (unfiltered acc)', color='lightblue')
axes_vel[0].plot(df_imu_plot['Time'], df_imu_plot['vx_median'], linewidth=2, label='Median Filter', color='cyan')
axes_vel[0].plot(df_imu_plot['Time'], df_imu_plot['vx_median_detrend'], linewidth=2.5, label='Median + Detrend', color='darkblue', linestyle='--')
axes_vel[0].set_ylabel('X Velocity (m/s)', fontsize=11)
axes_vel[0].legend(loc='upper right', fontsize=9)
axes_vel[0].grid(True, alpha=0.3)
axes_vel[0].set_title('X-axis Velocity (integrated from linear_acceleration.x)', loc='left', fontsize=10, fontweight='bold')
axes_vel[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Plot Y velocity
axes_vel[1].plot(df_imu_plot['Time'], df_imu_plot['vy_raw'], alpha=0.3, linewidth=0.8, label='Raw (unfiltered acc)', color='lightgreen')
axes_vel[1].plot(df_imu_plot['Time'], df_imu_plot['vy_median'], linewidth=2, label='Median Filter', color='cyan')
axes_vel[1].plot(df_imu_plot['Time'], df_imu_plot['vy_median_detrend'], linewidth=2.5, label='Median + Detrend', color='darkgreen', linestyle='--')
axes_vel[1].set_ylabel('Y Velocity (m/s)', fontsize=11)
axes_vel[1].legend(loc='upper right', fontsize=9)
axes_vel[1].grid(True, alpha=0.3)
axes_vel[1].set_title('Y-axis Velocity (integrated from linear_acceleration.y)', loc='left', fontsize=10, fontweight='bold')
axes_vel[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Plot Z velocity
axes_vel[2].plot(df_imu_plot['Time'], df_imu_plot['vz_raw'], alpha=0.3, linewidth=0.8, label='Raw (unfiltered acc)', color='lightcoral')
axes_vel[2].plot(df_imu_plot['Time'], df_imu_plot['vz_median'], linewidth=2, label='Median Filter', color='cyan')
axes_vel[2].plot(df_imu_plot['Time'], df_imu_plot['vz_median_detrend'], linewidth=2.5, label='Median + Detrend', color='darkred', linestyle='--')
axes_vel[2].set_ylabel('Z Velocity (m/s)', fontsize=11)
axes_vel[2].set_xlabel('Time (s)', fontsize=12)
axes_vel[2].legend(loc='upper right', fontsize=9)
axes_vel[2].grid(True, alpha=0.3)
axes_vel[2].set_title('Z-axis Velocity (integrated from linear_acceleration.z)', loc='left', fontsize=10, fontweight='bold')
axes_vel[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()

# Print statistics
print("\n" + "="*70)
print("                     ACCELERATION STATISTICS")
print("="*70)

print("\n### UKF WORLD-FRAME ACCELERATIONS ###")
print(f"\nX-acceleration (ax_world):")
print(f"  Mean: {df_ukf['ax_world'].mean():>8.4f} m/s²")
print(f"  Std:  {df_ukf['ax_world'].std():>8.4f} m/s²")
print(f"  Min:  {df_ukf['ax_world'].min():>8.4f} m/s²")
print(f"  Max:  {df_ukf['ax_world'].max():>8.4f} m/s²")

print(f"\nY-acceleration (ay_world):")
print(f"  Mean: {df_ukf['ay_world'].mean():>8.4f} m/s²")
print(f"  Std:  {df_ukf['ay_world'].std():>8.4f} m/s²")
print(f"  Min:  {df_ukf['ay_world'].min():>8.4f} m/s²")
print(f"  Max:  {df_ukf['ay_world'].max():>8.4f} m/s²")

print(f"\nZ-acceleration (az_world):")
print(f"  Mean: {df_ukf['az_world'].mean():>8.4f} m/s²")
print(f"  Std:  {df_ukf['az_world'].std():>8.4f} m/s²")
print(f"  Min:  {df_ukf['az_world'].min():>8.4f} m/s²")
print(f"  Max:  {df_ukf['az_world'].max():>8.4f} m/s²")

print(f"\nTotal samples: {len(df_ukf)}")
print(f"Duration: {df_ukf['t_s'].max():.3f} seconds")

print("\n" + "-"*70)
print("\n### RAW IMU LINEAR ACCELERATIONS ###")
print(f"\nX-acceleration (linear_acceleration.x):")
print(f"  Mean: {df_imu['linear_acceleration.x'].mean():>8.4f} m/s²")
print(f"  Std:  {df_imu['linear_acceleration.x'].std():>8.4f} m/s²")
print(f"  Min:  {df_imu['linear_acceleration.x'].min():>8.4f} m/s²")
print(f"  Max:  {df_imu['linear_acceleration.x'].max():>8.4f} m/s²")

print(f"\nY-acceleration (linear_acceleration.y):")
print(f"  Mean: {df_imu['linear_acceleration.y'].mean():>8.4f} m/s²")
print(f"  Std:  {df_imu['linear_acceleration.y'].std():>8.4f} m/s²")
print(f"  Min:  {df_imu['linear_acceleration.y'].min():>8.4f} m/s²")
print(f"  Max:  {df_imu['linear_acceleration.y'].max():>8.4f} m/s²")

print(f"\nZ-acceleration (linear_acceleration.z):")
print(f"  Mean: {df_imu['linear_acceleration.z'].mean():>8.4f} m/s²")
print(f"  Std:  {df_imu['linear_acceleration.z'].std():>8.4f} m/s²")
print(f"  Min:  {df_imu['linear_acceleration.z'].min():>8.4f} m/s²")
print(f"  Max:  {df_imu['linear_acceleration.z'].max():>8.4f} m/s²")

print(f"\nTotal samples: {len(df_imu)}")
print(f"Duration: {df_imu['Time'].max() - df_imu['Time'].min():.3f} seconds")

print("\n" + "-"*70)
print("\n### INTEGRATED VELOCITIES (from Median filtered acceleration) ###")
print(f"\nX-velocity (vx_median_detrend):")
print(f"  Mean: {df_imu['vx_median_detrend'].mean():>8.4f} m/s")
print(f"  Std:  {df_imu['vx_median_detrend'].std():>8.4f} m/s")
print(f"  Min:  {df_imu['vx_median_detrend'].min():>8.4f} m/s")
print(f"  Max:  {df_imu['vx_median_detrend'].max():>8.4f} m/s")

print(f"\nY-velocity (vy_median_detrend):")
print(f"  Mean: {df_imu['vy_median_detrend'].mean():>8.4f} m/s")
print(f"  Std:  {df_imu['vy_median_detrend'].std():>8.4f} m/s")
print(f"  Min:  {df_imu['vy_median_detrend'].min():>8.4f} m/s")
print(f"  Max:  {df_imu['vy_median_detrend'].max():>8.4f} m/s")

print(f"\nZ-velocity (vz_median_detrend):")
print(f"  Mean: {df_imu['vz_median_detrend'].mean():>8.4f} m/s")
print(f"  Std:  {df_imu['vz_median_detrend'].std():>8.4f} m/s")
print(f"  Min:  {df_imu['vz_median_detrend'].min():>8.4f} m/s")
print(f"  Max:  {df_imu['vz_median_detrend'].max():>8.4f} m/s")

print("\nNote: Velocities are computed via trapezoidal integration with linear detrending")
print("to remove drift. For accurate velocity estimation, proper bias correction is crucial.")
print("\n" + "="*70)

# ===== FREQUENCY ANALYSIS =====
print("\n### FREQUENCY ANALYSIS (Dominant Frequencies in IMU Data) ###")
print(f"\nSearching for dominant frequencies in range {DYNAMIC_NOTCH_RANGE[0]}-{DYNAMIC_NOTCH_RANGE[1]} Hz...")

freq_x = find_dominant_frequency(df_imu['linear_acceleration.x'].values, fs_imu, DYNAMIC_NOTCH_RANGE)
freq_y = find_dominant_frequency(df_imu['linear_acceleration.y'].values, fs_imu, DYNAMIC_NOTCH_RANGE)
freq_z = find_dominant_frequency(df_imu['linear_acceleration.z'].values, fs_imu, DYNAMIC_NOTCH_RANGE)

print(f"\nX-axis dominant frequency: {freq_x:.2f} Hz" if freq_x else "X-axis: No dominant frequency found")
print(f"Y-axis dominant frequency: {freq_y:.2f} Hz" if freq_y else "Y-axis: No dominant frequency found")
print(f"Z-axis dominant frequency: {freq_z:.2f} Hz" if freq_z else "Z-axis: No dominant frequency found")

print("\nNote: These frequencies likely correspond to motor vibrations or")
print("structural resonances. Dynamic notch filters track these frequencies.")
print("\n" + "="*70)

# ===== MEAN ACCELERATIONS IN TIME WINDOW =====
print("\n### MEAN ACCELERATIONS (Time window: 542-545 seconds) ###")

# Filter data for time window 542-545 seconds
time_mask = (df_imu['Time'] >= 542) & (df_imu['Time'] <= 545)
df_window = df_imu[time_mask]

if len(df_window) > 0:
    print(f"\nSamples in window: {len(df_window)}")
    print(f"Actual time range: {df_window['Time'].min():.3f} - {df_window['Time'].max():.3f} seconds")
    
    print("\nRAW LINEAR ACCELERATION (unfiltered):")
    print(f"  X-axis mean: {df_window['linear_acceleration.x'].mean():>8.4f} m/s²")
    print(f"  Y-axis mean: {df_window['linear_acceleration.y'].mean():>8.4f} m/s²")
    print(f"  Z-axis mean: {df_window['linear_acceleration.z'].mean():>8.4f} m/s²")
    
    print("\nMEDIAN FILTERED ACCELERATION:")
    print(f"  X-axis mean: {df_window['lax_median'].mean():>8.4f} m/s²")
    print(f"  Y-axis mean: {df_window['lay_median'].mean():>8.4f} m/s²")
    print(f"  Z-axis mean: {df_window['laz_median'].mean():>8.4f} m/s²")
    
    print("\nSTATIC NOTCH FILTERED ACCELERATION:")
    print(f"  X-axis mean: {df_window['lax_notch'].mean():>8.4f} m/s²")
    print(f"  Y-axis mean: {df_window['lay_notch'].mean():>8.4f} m/s²")
    print(f"  Z-axis mean: {df_window['laz_notch'].mean():>8.4f} m/s²")
    
    print("\nDYNAMIC NOTCH FILTERED ACCELERATION:")
    print(f"  X-axis mean: {df_window['lax_dynamic'].mean():>8.4f} m/s²")
    print(f"  Y-axis mean: {df_window['lay_dynamic'].mean():>8.4f} m/s²")
    print(f"  Z-axis mean: {df_window['laz_dynamic'].mean():>8.4f} m/s²")
else:
    print("\n⚠ WARNING: No data found in time window 542-545 seconds!")
    print(f"IMU data time range: {df_imu['Time'].min():.3f} - {df_imu['Time'].max():.3f} seconds")

print("\n" + "="*70)

# ===== CREATE FFT PLOTS =====
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle(f'Frequency Spectrum Analysis (IMU Data){interval_label}', fontsize=16, fontweight='bold')

for idx, (axis_name, axis_data, color) in enumerate([
    ('X', df_imu_plot['linear_acceleration.x'].values, 'blue'),
    ('Y', df_imu_plot['linear_acceleration.y'].values, 'green'),
    ('Z', df_imu_plot['linear_acceleration.z'].values, 'red')
]):
    # Compute FFT
    N = len(axis_data)
    yf = fft(axis_data)
    xf = fftfreq(N, 1/fs_imu)
    
    # Only plot positive frequencies
    pos_mask = xf > 0
    xf_pos = xf[pos_mask]
    yf_pos = np.abs(yf[pos_mask])
    
    # Plot frequency spectrum
    axes2[idx].semilogy(xf_pos, yf_pos, linewidth=1, color=color, alpha=0.7)
    axes2[idx].set_ylabel(f'{axis_name}-axis\nMagnitude', fontsize=11)
    axes2[idx].grid(True, alpha=0.3, which='both')
    axes2[idx].axvline(STATIC_NOTCH_FREQ, color='purple', linestyle='--', linewidth=2,
                       label=f'Static notch ({STATIC_NOTCH_FREQ} Hz)', alpha=0.7)
    
    # Mark dominant frequency
    dom_freq = [freq_x, freq_y, freq_z][idx]
    if dom_freq:
        axes2[idx].axvline(dom_freq, color='red', linestyle=':', linewidth=2.5,
                          label=f'Dominant freq ({dom_freq:.1f} Hz)', alpha=0.8)
    
    axes2[idx].legend(loc='upper right', fontsize=9)
    axes2[idx].set_xlim([0, 300])  # Focus on 0-300 Hz range
    axes2[idx].set_title(f'{axis_name}-axis Frequency Spectrum', loc='left', fontsize=10, fontweight='bold')

axes2[2].set_xlabel('Frequency (Hz)', fontsize=12)

plt.tight_layout()
plt.show()

print("\n✓ Plots generated successfully!")
print("\n" + "="*70)
print("                        FILTER RECOMMENDATIONS")
print("="*70)
print("\n1. ROLLING MEAN (Boxcar Filter):")
print("   - Simplest method, but allows high-frequency leakage")
print("   - Good for quick visualization, poor for precision control")
print("\n2. MEDIAN FILTER (Recommended for outlier removal):")
print("   - Non-linear filter excellent at removing outliers and spike noise")
print(f"   - Current config: kernel size = {MEDIAN_KERNEL}")
print("   - Preserves edges better than linear filters")
print("   - Ideal for: salt-and-pepper noise, impulsive disturbances")
print("   - Trade-off: Can introduce slight lag, not good for periodic signals")
print("\n3. STATIC NOTCH FILTER:")
print(f"   - Removes specific frequency: {STATIC_NOTCH_FREQ} Hz")
print("   - Good for constant motor idle frequency")
print("   - Limited effectiveness if motor RPM varies with throttle")
print("\n4. DYNAMIC NOTCH FILTER (Advanced):")
print("   - FFT-based tracking of dominant frequencies")
print(f"   - Searches in range: {DYNAMIC_NOTCH_RANGE[0]}-{DYNAMIC_NOTCH_RANGE[1]} Hz")
print("   - Tracks motor noise as throttle changes")
print("   - Used in Betaflight, ArduPilot, and modern flight controllers")
print("\n✓ Recommended Pipeline:")
print("  1. Median filter (remove outliers and spike noise)")
print("  2. Static notch (remove constant frequency noise)")
print("  3. Dynamic notch (if motor vibrations persist)")
print("\n" + "="*70)
print("                    VELOCITY INTEGRATION NOTES")
print("="*70)
print("\n✓ Integration Method: Trapezoidal rule (cumulative sum)")
print("  v(t) = v0 + ∫a(t)dt")
print("\n⚠ WARNING: Integration amplifies low-frequency errors!")
print("  - Small DC bias in acceleration → unbounded velocity drift")
print("  - Raw acceleration → unusable velocity (huge drift)")
print("  - Filtered acceleration → reduced drift but still present")
print("\n✓ Drift Correction Methods:")
print("  1. Linear detrending - Removes linear trend (assumes zero mean velocity)")
print("  2. High-pass filtering - Removes DC component")
print("  3. Periodic reset to ground truth (best for long sequences)")
print("\n✓ Best Practices:")
print("  - Use Median filtered acceleration for integration")
print("  - Apply detrending to remove systematic drift")
print("  - For position: Double integration makes drift even worse!")
print("  - Consider complementary filter with GPS/visual odometry")
print("="*70)
