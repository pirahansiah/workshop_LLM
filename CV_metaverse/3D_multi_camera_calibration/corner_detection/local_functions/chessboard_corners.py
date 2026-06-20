from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def chessboard_corners(img: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
    """Detect chessboard corners by scanning multiple board sizes.

    Iterates over a range of inner-corner counts to find the largest chessboard
    pattern present in *img*.  Returns the annotated image or ``None`` when no
    pattern is found.

    Args:
        img: BGR calibration image.

    Returns:
        Image with drawn corners, or ``None``.
    """
    max_a: int = 4
    max_b: int = 4
    found_points = False
    best_corners: NDArray[np.float32] | None = None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    for a in range(12, 35):
        for b in range(12, 35):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

            ret, corners = cv2.findChessboardCorners(gray, (a, b), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                if max_a < a and max_b < b:
                    max_a = a
                    max_b = b
                    best_corners = corners2
                    found_points = True
                print(f"a={a}, b={b} result {ret}  found")

    if found_points and best_corners is not None:
        cv2.drawChessboardCorners(img, (max_a, max_b), best_corners, True)
        return img

    return None
