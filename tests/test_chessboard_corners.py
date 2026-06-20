"""Tests for chessboard corner detection."""

from __future__ import annotations

import cv2
import numpy as np

from chessboard_corners import chessboard_corners


def _make_synthetic_chessboard(cols: int = 15, rows: int = 15, cell: int = 30) -> np.ndarray:
    """Create a synthetic chessboard image with enough inner corners."""
    w = cols * cell
    h = rows * cell
    board = np.zeros((h, w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def test_chessboard_corners_returns_image_on_valid_board() -> None:
    img = _make_synthetic_chessboard()
    result = chessboard_corners(img)
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_chessboard_corners_returns_none_on_blank_image() -> None:
    blank = np.zeros((400, 400, 3), dtype=np.uint8)
    result = chessboard_corners(blank)
    assert result is None
