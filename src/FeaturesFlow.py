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


def ORB_feature_matching(frame1, frame2, max_features=2000, match_ratio=0.75, use_cross_check=True):
    """
    Detect and match ORB features between two frames.
    
    Parameters:
    -----------
    frame1 : numpy array
        First grayscale frame
    frame2 : numpy array
        Second grayscale frame
    max_features : int
        Maximum number of ORB features to detect (default: 2000)
    match_ratio : float
        Ratio test threshold for Lowe's ratio test (default: 0.75)
        Lower values = more strict filtering
    use_cross_check : bool
        Whether to use cross-checking for matching (default: True)
    
    Returns:
    --------
    tuple : (pts1, pts2)
        pts1 : numpy array, shape (N, 1, 2)
            Matched keypoint coordinates in frame1
        pts2 : numpy array, shape (N, 1, 2)
            Corresponding matched keypoint coordinates in frame2
        Returns (None, None) if no good matches found
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)  # pylint: disable=no-member
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)
    
    # Check if descriptors were found
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None
    
    # Create BFMatcher
    if use_cross_check:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # pylint: disable=no-member
        matches = bf.match(des1, des2)
        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
    else:
        # Use ratio test (Lowe's ratio test)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # pylint: disable=no-member
        matches_knn = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        matches = []
        for m_n in matches_knn:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < match_ratio * n.distance:
                    matches.append(m)
    
    # Check if we have enough matches
    if len(matches) < 8:
        return None
    
    # Extract matched keypoint coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    return (pts1, pts2)