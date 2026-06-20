"""Corner detection and camera calibration using OpenCV.

References:
    1. Camera calibration for multi-modal robot vision based on image quality assessment
    2. Pattern image significance for camera calibration
    3. Camera Calibration and Video Stabilization Framework for Robot Localization
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def detect_corners_and_calibrate(
    gray_img: NDArray[np.uint8],
    checkerboard: tuple[int, int] = (6, 9),
) -> dict[str, object]:
    """Detect chessboard corners and compute camera calibration parameters.

    Args:
        gray_img: Grayscale calibration image.
        checkerboard: Inner corner dimensions (cols, rows).

    Returns:
        Dictionary with keys: ``camera_matrix``, ``dist_coeffs``,
        ``rotation_vectors``, ``translation_vectors``, ``corners_image``.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cols, rows = checkerboard
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints: list[NDArray[np.float32]] = []
    imgpoints: list[NDArray[np.float32]] = []

    corners_image = gray_img.copy()

    ret, corners = cv2.findChessboardCorners(
        gray_img,
        checkerboard,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if ret:
        corners_sub = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_sub)
        corners_image = cv2.drawChessboardCorners(gray_img, checkerboard, corners_sub, ret)

    mtx = dist = rvecs = tvecs = None
    ret_cal = False
    if objpoints and imgpoints:
        ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, corners_image.shape[::-1], None, None
        )

    return {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "rotation_vectors": rvecs,
        "translation_vectors": tvecs,
        "corners_image": corners_image,
        "calibration_success": ret_cal,
    }


def main() -> None:
    """Download a sample image and run corner detection + calibration."""
    import urllib.request
    from pathlib import Path

    url = (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/"
        "doc/tutorials/calib3d/camera_calibration/images/fileListImageUnDist.jpg"
    )

    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    local_path = cache_dir / "fileListImageUnDist.jpg"

    if not local_path.exists():
        urllib.request.urlretrieve(url, str(local_path))

    img_main = cv2.imread(str(local_path))
    if img_main is None:
        raise FileNotFoundError(f"Failed to read image: {local_path}")

    gray_img = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY) if len(img_main.shape) == 3 else img_main.copy()

    result = detect_corners_and_calibrate(gray_img)

    print("Camera matrix:\n", result["camera_matrix"])
    print("Distortion coefficients:\n", result["dist_coeffs"])
    print("Rotation vectors:\n", result["rotation_vectors"])
    print("Translation vectors:\n", result["translation_vectors"])


if __name__ == "__main__":
    main()
