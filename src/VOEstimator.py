import cv2
import logging
import numpy as np
from src.ConfigManager import ConfigManager
from src.DataManager import DataManager
from src.FeaturesFlow import detect_features, LK_optical_flow, ORB_feature_matching
from tqdm import tqdm

class VOEstimator:
    def __init__(self, dataset_path, calib_path, duration=None, use_imu=True, start_time=None):
        """
        Initialize Visual Odometry Estimator.
        
        Parameters:
        -----------
        dataset_path : str
            Path to dataset directory
        calib_path : str
            Path to calibration directory
        duration : float, optional
            Duration in seconds from start to process. If None, processes all data.
        """
        # Initialize any necessary variables or models here
        self.p0 = None
        self.camera_matrix = None  # You should set this to your camera's intrinsic parameters
        self.dist_coeffs = None  # Set this if you have lens distortion coefficients
        self.shift = None
        self.prev_frame = None
        self.curr_frame = None
        self.logger = logging.getLogger("VOEstimator")
        self.config_manager = ConfigManager(calib_path)
        self.data_manager = DataManager(dataset_path, duration=duration, use_imu=use_imu, start_time=start_time)
        pose, orientation = self.data_manager.get_first_init_groundtruth_pose()
        self.current_pose = pose
        self.current_orientation = orientation
        self.config = self.config_manager.config
        self.R = None
        self.shifts = {}
        self.setup_camera()

    def setup_camera(self):
        """Setup camera parameters from config."""
        self.camera_matrix = self.config_manager.get_camera_matrix()
        self.dist_coeffs = self.config_manager.get_distortion_coeffs()

    def undistort_image(self, image):
        K = self.camera_matrix
        D = self.dist_coeffs
        DIM = (image.shape[1], image.shape[0])

        # Create a new optimal camera matrix for fisheye
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2) # pylint: disable=no-member

        # Remap image
        undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) # pylint: disable=no-member
        return undistorted
    
    def derive_essential_matrix(self, p0_good, p1_good):
        # reshape
        x0 = p0_good.reshape(-1, 2)
        x1 = p1_good.reshape(-1, 2)

        # normalize using K^-1
        Kinv = np.linalg.inv(self.camera_matrix)

        x0 = Kinv @ np.hstack([x0, np.ones((x0.shape[0], 1))]).T
        x1 = Kinv @ np.hstack([x1, np.ones((x1.shape[0], 1))]).T

        # convert to Nx2 normalized coordinates
        x0 = (x0[:2] / x0[2]).T
        x1 = (x1[:2] / x1[2]).T

        # estimate essential matrix
        E, mask = cv2.findEssentialMat(
            x1,
            x0,
            focal=1.0,
            pp=(0, 0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.01
        )

        return E, mask, x0, x1


    def estimate_pose(self, p0_good, p1_good):
        E, mask, x0, x1 = self.derive_essential_matrix(p0_good, p1_good)
        _, R, t, mask_pose = cv2.recoverPose(E, x1, x0, focal=1.0, pp=(0, 0)) # pylint: disable=no-member
        t = np.array(t)
        t = self.current_orientation @ t.astype(np.float64)
        self.current_orientation = R @ self.current_orientation
        t = t.flatten()
        return t

    def update_attitude(self, imu_data):
        # self.R = imu_data
        pass

    def update(self, frame):
        frame = self.undistort_image(frame)
        if self.p0 is None:
            self.p0 = detect_features(frame)
        if self.prev_frame is None:
            self.prev_frame = frame
            return
        self.curr_frame = frame
        # tracked = LK_optical_flow(self.prev_frame, self.curr_frame, self.p0)
        tracked = ORB_feature_matching(self.prev_frame, self.curr_frame)
        self.prev_frame = self.curr_frame
        if tracked is None or len(tracked[1]) < 10:
            self.p0 = None
            return
        p0_good, p1_good = tracked
        self.p0 = p1_good.reshape(-1, 1, 2)
        self.shift = self.estimate_pose(p0_good, p1_good)

    def calculate_trajectory(self):
        estimated_positions = [self.shifts[t] for t in sorted(self.shifts.keys())]
        estimated_timestamps = list(self.shifts.keys())
        trajectory = np.cumsum(estimated_positions, axis=0) / 10 + self.current_pose
        return estimated_timestamps, trajectory

    def run(self):
        """
        Process all events from the dataset with a progress bar.
        """
        for event in tqdm(self.data_manager, desc="Processing events", unit="event"):
            event_type = event['type']
            event_time = event['time']
            
            if event_type == 'image':
                self.update(event['data']['image'])
                if self.shift is None:
                    self.shift = np.zeros(3)  # Initialize shift if not set
                self.shifts[event_time] = self.shift
            elif event_type == 'imu':
                self.update_attitude(event)

        return self.calculate_trajectory()