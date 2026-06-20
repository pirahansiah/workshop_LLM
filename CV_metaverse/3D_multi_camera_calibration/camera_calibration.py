"""Camera calibration for multi-camera 3D setups.

Uses OpenCV chessboard corner detection to compute intrinsics.
Run with a calibration image path argument, or uses a cached remote sample.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def calibrate_camera(
    image_path: str,
    checkerboard: tuple[int, int] = (6, 9),
) -> dict[str, object]:
    """Detect corners in *image_path* and calibrate the camera.

    Args:
        image_path: Filesystem path to a chessboard image.
        checkerboard: Inner corner grid (cols, rows).

    Returns:
        Dictionary with ``camera_matrix``, ``dist_coeffs``, ``rvecs``, ``tvecs``.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, rows = checkerboard

    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints: list[NDArray[np.float32]] = []
    imgpoints: list[NDArray[np.float32]] = []

    ret, corners = cv2.findChessboardCorners(
        gray,
        checkerboard,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    corners_img = gray.copy()
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_sub)
        corners_img = cv2.drawChessboardCorners(gray, checkerboard, corners_sub, ret)

    mtx = dist = rvecs = tvecs = None
    ret_cal = False
    if objpoints and imgpoints:
        ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, corners_img.shape[::-1], None, None
        )

    print("Camera matrix:\n", mtx)
    print("Distortion:\n", dist)
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)

    return {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "rotation_vectors": rvecs,
        "translation_vectors": tvecs,
        "calibration_success": ret_cal,
    }


if __name__ == "__main__":
    import sys
    import urllib.request
    from pathlib import Path

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        url = (
            "https://raw.githubusercontent.com/opencv/opencv/4.x/"
            "doc/tutorials/calib3d/camera_calibration/images/fileListImageUnDist.jpg"
        )
        cache_dir = Path(__file__).resolve().parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        image_path = str(cache_dir / "sample.jpg")
        if not Path(image_path).exists():
            urllib.request.urlretrieve(url, image_path)
        print(f"No path argument — using cached sample: {image_path}")

    calibrate_camera(image_path)
