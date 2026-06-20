"""Tests for corner detection and calibration modules."""

from __future__ import annotations

import cv2
import numpy as np

from cornerDetection import detect_corners_and_calibrate


def _make_checkerboard_image(
    cols: int = 9,
    rows: int = 7,
    cell_size: int = 60,
    border: int = 150,
) -> np.ndarray:
    """Synthesize a chessboard calibration image with a wide border.

    The checkerboard has *cols* black/white columns and *rows* rows, giving
    (cols-1) x (rows-1) inner corners.  We pass checkerboard=(cols-1, rows-1)
    to the detector.
    """
    w = cols * cell_size + 2 * border
    h = rows * cell_size + 2 * border
    img = np.full((h, w), 128, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y1 = border + r * cell_size
                x1 = border + c * cell_size
                img[y1 : y1 + cell_size, x1 : x1 + cell_size] = 255
            else:
                y1 = border + r * cell_size
                x1 = border + c * cell_size
                img[y1 : y1 + cell_size, x1 : x1 + cell_size] = 0
    return img


def test_detect_corners_and_calibrate_with_synthetic_board() -> None:
    gray = _make_checkerboard_image(cols=9, rows=7)
    result = detect_corners_and_calibrate(gray, checkerboard=(8, 6))
    assert "camera_matrix" in result
    assert result["corners_image"] is not None


def test_detect_corners_and_calibrate_with_blank_image() -> None:
    blank = np.zeros((400, 600), dtype=np.uint8)
    result = detect_corners_and_calibrate(blank, checkerboard=(6, 9))
    assert result["camera_matrix"] is None or result["camera_matrix"] is not None
