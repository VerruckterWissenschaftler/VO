"""
Standalone viewer for dvs-image_raw.csv frames.

Usage
-----
    python view_frames.py                         # all frames, default path
    python view_frames.py --start 1540113200.0    # start at absolute timestamp
    python view_frames.py --start 1540113200.0 --duration 5.0
    python view_frames.py --csv path/to/dvs-image_raw.csv --start 0 --duration 10

Controls (OpenCV window)
------------------------
    Space / any key  — next frame
    Esc / q          — quit
"""

import argparse
import sys
import cv2
import numpy as np
import pandas as pd
import ast


# ── CSV parsing ──────────────────────────────────────────────────────────────

def _parse_image(row) -> dict:
    height   = int(row["height"])
    width    = int(row["width"])
    encoding = str(row["encoding"]).lower()
    data_str = str(row["data"]).strip()

    if encoding not in ("mono8", "8uc1"):
        raise ValueError(f"Unsupported encoding: {encoding}")
    if not (data_str.startswith("b'") or data_str.startswith('b"')):
        raise ValueError("data field is not a byte string literal")

    img_bytes = ast.literal_eval(data_str)
    img_flat  = np.frombuffer(img_bytes, dtype=np.uint8)

    if img_flat.size != height * width:
        raise ValueError(f"Size mismatch: got {img_flat.size}, expected {height*width}")

    return {"time": float(row["Time"]), "image": img_flat.reshape((height, width))}


def iter_frames(csv_path: str, start: float | None, duration: float | None):
    """Yield parsed frame dicts filtered by [start, start+duration)."""
    df = pd.read_csv(csv_path)

    t0 = start if start is not None else float(df["Time"].iloc[0])
    t1 = (t0 + duration) if duration is not None else float("inf")

    for _, row in df.iterrows():
        t = float(row["Time"])
        if t < t0:
            continue
        if t >= t1:
            break
        yield _parse_image(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View DAVIS DVS image frames")
    parser.add_argument(
        "--csv",
        default=r"data\outdoor_forward\outdoor_forward_1_davis_with_gt\dvs-image_raw.csv",
        help="Path to dvs-image_raw.csv",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Absolute timestamp to begin playback (default: first frame)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds of frames to show (default: until end of file)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Display scale factor (default: 2.0)",
    )
    args = parser.parse_args()

    window = "DVS Frames  [Space=next  Esc/q=quit]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    for frame in iter_frames(args.csv, args.start, args.duration):
        img      = frame["image"]
        t        = frame["time"]
        display  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Overlay timestamp and frame index
        label = f"t={t:.4f}  frame={frame_idx}"
        cv2.putText(display, label, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Scale up for visibility
        h, w = display.shape[:2]
        display = cv2.resize(display, (int(w * args.scale), int(h * args.scale)),
                             interpolation=cv2.INTER_NEAREST)

        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):   # Esc or q
            break

        frame_idx += 1

    cv2.destroyAllWindows()
    print(f"Displayed {frame_idx} frames.")


if __name__ == "__main__":
    main()
