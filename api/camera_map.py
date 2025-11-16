# -----------------------------------------------------------------------------
# Copyright (c) 2025
#
# Authors:
#   Liruo Wang
#       School of Electrical Engineering and Computer Science,
#       University of Ottawa
#       lwang032@uottawa.ca
#
# All rights reserved.
# -----------------------------------------------------------------------------

# in real world,one camera can transfer only one video per time,so its natural to be one camera_id for one video source
import glob

BASE_DIR = r"E:\Training\traffic_video"

def find_video_for_cam(cam_id: int):
    """
    Given cam_id = 1 → find file containing '-cam-1.mp4'
    """
    pattern = fr"{BASE_DIR}\*-cam-{cam_id}.mp4"
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No video file found for cam-{cam_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple video files match cam-{cam_id}: {matches}")
    return matches[0]

CAMERA_SOURCES = {
    f"cam-{i}": {
        "src": find_video_for_cam(i),
        "enabled": True
    }
    for i in range(1, 41)
}


def get_source(camera_id: str) -> str:
    meta = CAMERA_SOURCES.get(camera_id)
    if not meta or not meta.get("enabled", True):
        raise KeyError(f"camera_id is not configured or is disabled: {camera_id}")
    src = (meta.get("src") or "").strip()
    if not src:
        raise ValueError(f"camera_id does not provide a valid src: {camera_id}")
    return src

