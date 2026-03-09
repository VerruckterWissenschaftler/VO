import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Classical CPU matchers
# ---------------------------------------------------------------------------

def detect_features(frame):
    p0 = cv2.goodFeaturesToTrack(  # pylint: disable=no-member
        frame,
        maxCorners=1000,
        qualityLevel=0.1,
        minDistance=20,
        blockSize=7,
    )
    return p0


def _lk_track(frame1, frame2, prev_pts):
    """LK tracking with forward-backward consistency check. Returns (p0, p1) or None."""
    p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, prev_pts, None)  # pylint: disable=no-member
    p0_est, st_bwd, _ = cv2.calcOpticalFlowPyrLK(frame2, frame1, p1, None)  # pylint: disable=no-member
    fb_err = np.linalg.norm(prev_pts - p0_est, axis=2).reshape(-1)
    good = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1) & (fb_err < 1.0)
    if good.sum() < 8:
        return None
    return prev_pts[good], p1[good]


def LK_optical_flow(prev_frame, curr_frame, p0):
    """Lucas-Kanade optical flow with forward-backward consistency check."""
    if p0 is None:
        p0 = detect_features(prev_frame)
    if p0 is None:
        return None
    return _lk_track(prev_frame, curr_frame, p0)


class StatefulTracker:
    """
    Detect-and-track matcher for SIFT / ORB / Shi-Tomasi corners.

    Detects features on the first call (or when tracked count drops below
    *min_tracked*), then propagates them frame-to-frame via LK optical flow.
    Feature detection cost is paid only ~once per *min_tracked* interval instead
    of every frame.
    """

    _LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))  # pylint: disable=no-member

    def __init__(self, detector: str = "shi-tomasi", max_features: int = 1000,
                 min_tracked: int = 200):
        self.detector = detector
        self.max_features = max_features
        self.min_tracked = min_tracked
        self._prev_pts: np.ndarray | None = None  # (N,1,2) float32 in prev_frame coords

    # ------------------------------------------------------------------
    def __call__(self, frame1: np.ndarray, frame2: np.ndarray):
        """Returns (p0, p1) both shape (N,1,2), or None if insufficient."""
        if self._prev_pts is None or len(self._prev_pts) < self.min_tracked:
            self._prev_pts = self._detect(frame1)

        if self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_pts = None
            return None

        result = _lk_track(frame1, frame2, self._prev_pts)
        if result is None:
            self._prev_pts = None
            return None

        p0_good, p1_good = result
        # Surviving tracked points in frame2 become prev_pts for next call
        self._prev_pts = p1_good.reshape(-1, 1, 2).copy()
        return p0_good.reshape(-1, 1, 2), p1_good.reshape(-1, 1, 2)

    def reset(self):
        self._prev_pts = None

    # ------------------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> np.ndarray | None:
        if self.detector == "sift":
            sift = cv2.SIFT_create(nfeatures=self.max_features)  # pylint: disable=no-member
            kps = sift.detect(frame, None)
            if not kps:
                return None
            return np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)

        if self.detector == "orb":
            orb = cv2.ORB_create(  # pylint: disable=no-member
                nfeatures=self.max_features, scaleFactor=1.2, nlevels=8,
                edgeThreshold=31, scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=20,  # pylint: disable=no-member
            )
            kps = orb.detect(frame, None)
            if not kps:
                return None
            return np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)

        # Default: Shi-Tomasi / goodFeaturesToTrack
        return cv2.goodFeaturesToTrack(  # pylint: disable=no-member
            frame, maxCorners=self.max_features, qualityLevel=0.1,
            minDistance=20, blockSize=7,
        )


def ORB_feature_matching(frame1, frame2, max_features=2000):
    orb = cv2.ORB_create(  # pylint: disable=no-member
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        scoreType=cv2.ORB_HARRIS_SCORE,
        fastThreshold=20,
    )
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # pylint: disable=no-member
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    matches = [m for m in matches if m.distance < 40]

    if len(matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return (pts1, pts2)


def SIFT_feature_matching(frame1, frame2, max_features=2000, match_ratio=0.75, use_flann=True):
    """SIFT with Lowe's ratio test. Active default matcher."""
    sift = cv2.SIFT_create(nfeatures=max_features)  # pylint: disable=no-member
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None

    if use_flann:
        index_params = dict(algorithm=1, trees=5)
        matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))  # pylint: disable=no-member
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # pylint: disable=no-member

    matches_knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for m_n in matches_knn if len(m_n) == 2
            for m, n in [m_n] if m.distance < match_ratio * n.distance]

    if len(good) < 8:
        return None

    p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return (p1, p2)

# ---------------------------------------------------------------------------
# Matcher selector
# ---------------------------------------------------------------------------

def get_matcher(name: str = "sift"):
    """
    Return a feature matching function by name.

    Supported names: 'sift', 'orb', 'lk', 'lightglue', 'loftr'

    sift / orb / lk  →  StatefulTracker  (detect once, track via LK until
                         point count drops below threshold, then re-detect)
    lightglue / loftr → GPU descriptor matchers (detect on every frame pair)

    The returned callable has signature:
        matcher(frame1, frame2) → (p1, p2) | None
    """
    name_lower = name.lower()
    if name_lower == "sift":
        return StatefulTracker(detector="sift", max_features=1000, min_tracked=200)
    if name_lower == "orb":
        return StatefulTracker(detector="orb", max_features=2000, min_tracked=200)
    if name_lower == "lk":
        return StatefulTracker(detector="shi-tomasi", max_features=1000, min_tracked=200)
    raise ValueError(f"Unknown matcher '{name}'. Choose from: sift, orb, lk, lightglue, loftr")
