import cv2
import numpy as np

def detect_features(frame):
    p0 = cv2.goodFeaturesToTrack( # pylint: disable=no-member
        frame,
        maxCorners=1000,
        qualityLevel=0.1,
        minDistance=5,
        blockSize=7
    )
    return p0

def LK_optical_flow(prev_frame, curr_frame, p0):
    # Implement Lucas-Kanade optical flow to track features between frames
    if p0 is None:
        p0 = detect_features(prev_frame)
        return None
    
    p1, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK( # pylint: disable=no-member
        prev_frame, curr_frame, p0, None
    )

    p0_est, st_bwd, err_bwd = cv2.calcOpticalFlowPyrLK( # pylint: disable=no-member
        curr_frame, prev_frame, p1, None
    )

    fb_err = np.linalg.norm(p0 - p0_est, axis=2).reshape(-1)
    good = (fb_err < 1.0)

    p0_good = p0[good]
    p1_good = p1[good]

    return (p0_good, p1_good)