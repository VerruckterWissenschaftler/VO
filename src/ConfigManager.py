import yaml
import os
import numpy as np


class ConfigManager:
    def __init__(self, calib_dir="data/outdoor_forward_calib_davis"):
        """
        Initialize ConfigManager by loading calibration YAML files.
        
        Parameters:
        -----------
        calib_dir : str
            Path to the directory containing calibration YAML files
        """
        self.calib_dir = calib_dir
        self.config = self.load_all_configs()
    
    def load_yaml(self, filepath):
        """Load a single YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def load_all_configs(self):
        """
        Load all calibration YAML files from the calibration directory.
        
        Returns:
        --------
        config : dict
            Dictionary containing all calibration data with keys:
            - 'imu': IMU calibration parameters
            - 'camera': Camera calibration parameters
            - 'camchain': Camera-IMU transformation chain
            - 'target': Calibration target parameters
        """
        config = {}
        
        # Load IMU calibration
        imu_file = os.path.join(self.calib_dir, "imu-..outdoor_forward_calib_davis_imu.yaml")
        if os.path.exists(imu_file):
            imu_data = self.load_yaml(imu_file)
            config['imu'] = imu_data.get('imu0', {})
        
        # Load camera-IMU chain calibration
        camchain_file = os.path.join(self.calib_dir, "camchain-imucam-..outdoor_forward_calib_davis_imu.yaml")
        if os.path.exists(camchain_file):
            camchain_data = self.load_yaml(camchain_file)
            config['camchain'] = camchain_data
            config['camera'] = camchain_data.get('cam0', {})
        
        # Load calibration target parameters
        target_file = os.path.join(self.calib_dir, "target.yaml")
        if os.path.exists(target_file):
            config['target'] = self.load_yaml(target_file)
        
        return config
    
    def get_camera_matrix(self):
        """
        Get camera intrinsic matrix K.
        
        Returns:
        --------
        K : numpy.ndarray, shape (3, 3)
            Camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        intrinsics = self.config['camera']['intrinsics']
        # intrinsics format: [fx, fy, cx, cy]
        fx, fy, cx, cy = intrinsics
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    def get_distortion_coeffs(self):
        """
        Get camera distortion coefficients.
        
        Returns:
        --------
        dist_coeffs : numpy.ndarray
            Distortion coefficients [k1, k2, k3, k4] for equidistant (fisheye) model
        """
        return np.array(self.config['camera']['distortion_coeffs'], dtype=np.float32)
    
    def get_camera_resolution(self):
        """
        Get camera resolution.
        
        Returns:
        --------
        resolution : tuple
            (width, height)
        """
        res = self.config['camera']['resolution']
        return tuple(res)  # (width, height)
    
    def get_T_cam_imu(self):
        """
        Get transformation matrix from IMU frame to camera frame.
        
        Returns:
        --------
        T_cam_imu : numpy.ndarray, shape (4, 4)
            Transformation matrix from IMU to camera coordinate system
        """
        return np.array(self.config['camera']['T_cam_imu'], dtype=np.float32)
    
    def get_T_i_b(self):
        """
        Get transformation matrix from body frame to IMU frame.
        
        Returns:
        --------
        T_i_b : numpy.ndarray, shape (4, 4)
            Transformation matrix from body to IMU coordinate system
        """
        return np.array(self.config['imu']['T_i_b'], dtype=np.float32)
    
    def get_imu_params(self):
        """
        Get IMU noise parameters.
        
        Returns:
        --------
        params : dict
            Dictionary containing:
            - accelerometer_noise_density
            - accelerometer_random_walk
            - gyroscope_noise_density
            - gyroscope_random_walk
            - update_rate
            - time_offset
        """
        imu_config = self.config['imu']
        return {
            'accelerometer_noise_density': imu_config['accelerometer_noise_density'],
            'accelerometer_random_walk': imu_config['accelerometer_random_walk'],
            'gyroscope_noise_density': imu_config['gyroscope_noise_density'],
            'gyroscope_random_walk': imu_config['gyroscope_random_walk'],
            'update_rate': imu_config['update_rate'],
            'time_offset': imu_config['time_offset']
        }
    
    def get_timeshift_cam_imu(self):
        """
        Get time shift between camera and IMU.
        
        Returns:
        --------
        timeshift : float
            Time shift in seconds
        """
        return self.config['camera']['timeshift_cam_imu']
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration."""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        
        print("\nCamera:")
        print(f"  Model: {self.config['camera'].get('camera_model', 'N/A')}")
        print(f"  Resolution: {self.get_camera_resolution()}")
        print(f"  Intrinsics (fx, fy, cx, cy): {self.config['camera']['intrinsics']}")
        print(f"  Distortion model: {self.config['camera'].get('distortion_model', 'N/A')}")
        
        print("\nIMU:")
        print(f"  Model: {self.config['imu'].get('model', 'N/A')}")
        print(f"  Update rate: {self.config['imu']['update_rate']} Hz")
        print(f"  Accel noise density: {self.config['imu']['accelerometer_noise_density']}")
        print(f"  Gyro noise density: {self.config['imu']['gyroscope_noise_density']}")
        
        print("\nSynchronization:")
        print(f"  Time shift (cam-imu): {self.get_timeshift_cam_imu():.6f} s")
        
        if 'target' in self.config:
            print("\nCalibration Target:")
            print(f"  Type: {self.config['target'].get('target_type', 'N/A')}")
            if 'tagCols' in self.config['target']:
                print(f"  Grid: {self.config['target']['tagRows']}x{self.config['target']['tagCols']}")
                print(f"  Tag size: {self.config['target']['tagSize']} m")
        
        print("=" * 60)
