"""Tests for the 3D multi-camera calibration module."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "cal3d",
    Path(__file__).resolve().parent.parent
    / "CV_metaverse" / "3D_multi_camera_calibration" / "camera_calibration.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
calibrate_camera = _mod.calibrate_camera


def _make_calibration_image(path: str) -> None:
    cell, border = 60, 150
    cols, rows = 9, 7
    w = cols * cell + 2 * border
    h = rows * cell + 2 * border
    img = np.full((h, w), 128, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y1 = border + r * cell
            x1 = border + c * cell
            if (r + c) % 2 == 0:
                img[y1 : y1 + cell, x1 : x1 + cell] = 255
            else:
                img[y1 : y1 + cell, x1 : x1 + cell] = 0
    cv2.imwrite(path, img)


def test_calibrate_camera_valid(tmp_path) -> None:
    img_path = str(tmp_path / "board.jpg")
    _make_calibration_image(img_path)
    result = calibrate_camera(img_path, checkerboard=(8, 6))
    assert "camera_matrix" in result


def test_calibrate_camera_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        calibrate_camera("/nonexistent/image.jpg")
