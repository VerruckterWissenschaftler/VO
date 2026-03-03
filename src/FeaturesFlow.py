import cv2
import numpy as np
import torch
from pathlib import Path


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


def LK_optical_flow(prev_frame, curr_frame, p0):
    """Lucas-Kanade optical flow with forward-backward consistency check."""
    if p0 is None:
        detect_features(prev_frame)
        return None

    p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, p0, None)  # pylint: disable=no-member
    p0_est, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_frame, prev_frame, p1, None)  # pylint: disable=no-member

    fb_err = np.linalg.norm(p0 - p0_est, axis=2).reshape(-1)
    good = fb_err < 1.0

    return (p0[good], p1[good])


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
# GPU matchers — loaded lazily (models cached after first call)
# ---------------------------------------------------------------------------

_lightglue_cache: dict = {}   # device → (extractor, matcher)
_loftr_cache: dict = {}       # device → matcher


def LightGlue_matching(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_features: int = 2048,
    device: str | None = None,
):
    """
    LightGlue feature matching using SuperPoint keypoints.

    Requires: pip install git+https://github.com/cvg/LightGlue.git

    Parameters
    ----------
    frame1, frame2 : uint8 grayscale images (H, W)
    max_features   : maximum SuperPoint keypoints per image
    device         : 'cuda' / 'cpu' / None (auto-detect)

    Returns
    -------
    (p1, p2) arrays of shape (N, 1, 2) or None if insufficient matches.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global _lightglue_cache
    if device not in _lightglue_cache:
        from lightglue import LightGlue, SuperPoint  # lazy import
        extractor = SuperPoint(max_num_keypoints=max_features).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        _lightglue_cache[device] = (extractor, matcher)

    extractor, matcher = _lightglue_cache[device]

    # Prepare tensors: float32 [0,1], shape [1, 1, H, W]
    def _to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)

    img0, img1 = _to_tensor(frame1), _to_tensor(frame2)

    with torch.no_grad():
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        result = matcher({"image0": feats0, "image1": feats1})

    # Remove batch dimension
    from lightglue.utils import rbd
    feats0, feats1, result = rbd(feats0), rbd(feats1), rbd(result)

    kpts0 = feats0["keypoints"].cpu().numpy()   # (N, 2)
    kpts1 = feats1["keypoints"].cpu().numpy()   # (M, 2)
    matches = result["matches"].cpu().numpy()   # (K, 2)

    if len(matches) < 8:
        return None

    p1 = kpts0[matches[:, 0]].reshape(-1, 1, 2).astype(np.float32)
    p2 = kpts1[matches[:, 1]].reshape(-1, 1, 2).astype(np.float32)
    return (p1, p2)


def LOFTR_matching(
    frame1: np.ndarray,
    frame2: np.ndarray,
    device: str | None = None,
):
    """
    LoFTR detector-free dense feature matching via kornia.

    Requires: pip install kornia

    Parameters
    ----------
    frame1, frame2 : uint8 grayscale images (H, W)
    device         : 'cuda' / 'cpu' / None (auto-detect)

    Returns
    -------
    (p1, p2) arrays of shape (N, 1, 2) or None if insufficient matches.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global _loftr_cache
    if device not in _loftr_cache:
        import kornia.feature as KF  # lazy import
        _loftr_cache[device] = KF.LoFTR(pretrained="outdoor").eval().to(device)

    loftr = _loftr_cache[device]

    def _to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)

    img0, img1 = _to_tensor(frame1), _to_tensor(frame2)

    with torch.no_grad():
        out = loftr({"image0": img0, "image1": img1})

    kpts0 = out["keypoints0"].cpu().numpy()  # (M, 2)
    kpts1 = out["keypoints1"].cpu().numpy()  # (M, 2)

    if len(kpts0) < 8:
        return None

    p1 = kpts0.reshape(-1, 1, 2).astype(np.float32)
    p2 = kpts1.reshape(-1, 1, 2).astype(np.float32)
    return (p1, p2)


# ---------------------------------------------------------------------------
# Matcher selector
# ---------------------------------------------------------------------------

def get_matcher(name: str = "sift"):
    """
    Return a feature matching function by name.

    Supported names: 'sift', 'orb', 'lk', 'lightglue', 'loftr'

    The returned callable has signature:
        matcher(frame1, frame2) → (p1, p2) | None
    """
    _registry = {
        "sift": SIFT_feature_matching,
        "orb": ORB_feature_matching,
        "lk": lambda f1, f2: LK_optical_flow(f1, f2, p0=None),
        "lightglue": LightGlue_matching,
        "loftr": LOFTR_matching,
    }
    name_lower = name.lower()
    if name_lower not in _registry:
        raise ValueError(
            f"Unknown matcher '{name}'. Choose from: {list(_registry.keys())}"
        )
    return _registry[name_lower]
