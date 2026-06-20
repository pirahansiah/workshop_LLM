"""Pytest configuration — adds source directories to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent

# Add paths for import resolution
_corner = _root / "CV_metaverse" / "3D_multi_camera_calibration" / "corner_detection"
_local = _corner / "local_functions"
_cal3d = _root / "CV_metaverse" / "3D_multi_camera_calibration"

for p in (_local, _corner, _cal3d):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
