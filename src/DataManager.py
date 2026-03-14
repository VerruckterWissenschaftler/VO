import pandas as pd
from src.DavisCsvReader import DavisCsvReader
import numpy as np
import os


class DataManager:
    def __init__(self, dataset_path, duration=float('inf'), use_imu=False, start_time=None):
        self.dataset_path = dataset_path
        self.duration = duration
        
        # Read ground truth data first to determine time bounds
        self.gt_df = pd.read_csv(os.path.join(dataset_path, "groundtruth-pose.csv"))
        
        # Read IMU data if requested
        self.imu_df = None
        if use_imu:
            self.imu_df = pd.read_csv(os.path.join(dataset_path, "dvs-imu.csv"))
        
        # Use DavisCsvReader for image data (handles image decoding)
        self.image_reader = DavisCsvReader(os.path.join(dataset_path, "dvs-image_raw.csv"))
        
        # Store the start time (reference time from image data or provided)
        if start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = self.image_reader.df['Time'].min() if len(self.image_reader.df) > 0 else 0.0
        
        # Determine end time from ground truth
        if len(self.gt_df) > 0:
            gt_end_time = self.gt_df['Time'].max()
            # Use the minimum of specified duration and ground truth end time
            calculated_end_time = self.start_time + self.duration
            self.end_time = min(calculated_end_time, gt_end_time)
            print(f"Ground truth end time: {gt_end_time:.6f}")
            print(f"Using end time: {self.end_time:.6f} (limited by {'GT' if self.end_time == gt_end_time else 'duration'})")
        else:
            self.end_time = self.start_time + self.duration
            print("Warning: No ground truth data available")
        
        # Apply duration filter if specified
        self._apply_duration_filter()
        
        # Build the event list
        self.events = self._build_event_list()
        self._current_index = 0  # For iteration
    
    def get_first_init_groundtruth_pose(self):
        if len(self.gt_df) == 0:
            return None, None
        
        first_row = self.gt_df.iloc[0]
        init_position = np.array([
            first_row["pose.position.x"],
            first_row["pose.position.y"],
            first_row["pose.position.z"]
        ])

        q = np.array([
            first_row["pose.orientation.x"],
            first_row["pose.orientation.y"],
            first_row["pose.orientation.z"],
            first_row["pose.orientation.w"]
        ]).astype(float)

        # unpack
        x, y, z, w = q

        norm = np.linalg.norm(q)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

        
        return init_position, R

    def _apply_duration_filter(self):
        """
        Filter data to only include events within the time window.
        Filters IMU data, image data, and ground truth to the time window
        [start_time, end_time].
        End time is determined by the minimum of (start_time + duration) and 
        the last ground truth timestamp.
        """
        # Use precalculated end_time
        start_time = self.start_time
        end_time = self.end_time
        
        actual_duration = end_time - start_time
        print(f"\nApplying time filter: {actual_duration:.3f}s")
        print(f"Time window: [{start_time:.6f}, {end_time:.6f}]")
        
        # Filter IMU data
        if self.imu_df is not None:
            imu_before = len(self.imu_df)
            self.imu_df = self.imu_df[
                (self.imu_df['Time'] >= start_time) & 
                (self.imu_df['Time'] <= end_time)
            ].reset_index(drop=True)
            print(f"IMU events: {imu_before} -> {len(self.imu_df)}")
        
        # Filter image data
        img_before = len(self.image_reader.df)
        self.image_reader.df = self.image_reader.df[
            (self.image_reader.df['Time'] >= start_time) & 
            (self.image_reader.df['Time'] <= end_time)
        ].reset_index(drop=True)
        print(f"Image events: {img_before} -> {len(self.image_reader.df)}")
        
        # Filter ground truth data
        gt_before = len(self.gt_df)
        self.gt_df = self.gt_df[
            (self.gt_df['Time'] >= start_time) & 
            (self.gt_df['Time'] <= end_time)
        ].reset_index(drop=True)
        print(f"Ground truth poses: {gt_before} -> {len(self.gt_df)}")
        print()
    
    def _build_event_list(self):
        """
        Build a sorted list of events from IMU and image data.
        Returns a list of dictionaries with event information.
        """
        events = []
        
        # Add IMU events
        if self.imu_df is not None:
            for idx, row in self.imu_df.iterrows():
                events.append({
                    'time': row['Time'],
                    'type': 'imu',
                    'data': row.to_dict()
                })
        
        # Add image events (only metadata, not the actual image yet)
        image_df = self.image_reader.df
        for idx, row in image_df.iterrows():
            events.append({
                'time': row['Time'],
                'type': 'image',
                'data': {
                    'Time': row['Time'],
                    'height': int(row['height']),
                    'width': int(row['width']),
                    '_row_index': idx  # position in filtered df for lazy loading
                }
            })
        
        # Sort all events by time
        events.sort(key=lambda x: x['time'])
        
        return events
    
    def get_image_at_index(self, row_index):
        row = self.image_reader.df.iloc[row_index]
        return self.image_reader._parse_row(row)
    
    def get_events(self):
        # Convert events list to DataFrame for compatibility
        df_data = []
        for event in self.events:
            row = {'time': event['time'], 'type': event['type']}
            # Flatten the data dictionary
            for key, val in event['data'].items():
                if key != 'data' and key != '_row_index':  # Skip raw image data
                    row[key] = val
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def get_groundtruth_trajectory(self):
        """
        Extract the ground truth trajectory as timestamps and positions.
        
        Returns:
        --------
        timestamps : numpy array
            Timestamps from ground truth data (original timestamps from dataset)
        positions : numpy array, shape (N, 3)
            [x, y, z] positions from ground truth
        orientations : numpy array, shape (N, 4)
            [qx, qy, qz, qw] quaternion orientations from ground truth
            
        Notes:
        ------
        - Ground truth timestamps are NOT adjusted/synchronized
        - When duration filter is applied, only GT within the time window is returned
        - If no ground truth data exists, returns empty arrays with correct shapes
        """
        if len(self.gt_df) == 0:
            # Return empty arrays if no ground truth data
            return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 4)
        
        timestamps = self.gt_df["Time"].values
        positions = self.gt_df[["pose.position.x", "pose.position.y", "pose.position.z"]].values
        orientations = self.gt_df[["pose.orientation.x", "pose.orientation.y", 
                                    "pose.orientation.z", "pose.orientation.w"]].values
        
        return timestamps, positions, orientations
    
    def __iter__(self):
        """Initialize iterator to start from the beginning of events."""
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self.events):
            raise StopIteration
        
        event = self.events[self._current_index].copy()
        
        # For image events, decode the image lazily
        if event['type'] == 'image' and '_row_index' in event['data']:
            row_index = event['data']['_row_index']
            image_data = self.get_image_at_index(row_index)
            event['data'] = image_data
        
        self._current_index += 1
        return event
    
    def __len__(self):
        """Return the total number of events."""
        return len(self.events)
    
    def print_time_alignment_info(self):
        """
        Print information about time alignment across all data sources.
        Useful for debugging time synchronization issues.
        """
        print("\n" + "="*60)
        print("Time Alignment Information")
        print("="*60)
        
        if len(self.image_reader.df) > 0:
            img_start = self.image_reader.df['Time'].min()
            img_end = self.image_reader.df['Time'].max()
            print(f"Image data:  [{img_start:.6f}, {img_end:.6f}]  ({img_end - img_start:.3f}s)")
        else:
            print("Image data:  No data")
        if self.imu_df is not None and len(self.imu_df) > 0:
            imu_start = self.imu_df['Time'].min()
            imu_end = self.imu_df['Time'].max()
            print(f"IMU data:    [{imu_start:.6f}, {imu_end:.6f}]  ({imu_end - imu_start:.3f}s)")
        else:
            print("IMU data:    No data")
    
        if len(self.gt_df) > 0:
            gt_start = self.gt_df['Time'].min()
            gt_end = self.gt_df['Time'].max()
            print(f"GT data:     [{gt_start:.6f}, {gt_end:.6f}]  ({gt_end - gt_start:.3f}s)")
        else:
            print("GT data:     No data")
        
        print("="*60 + "\n")
