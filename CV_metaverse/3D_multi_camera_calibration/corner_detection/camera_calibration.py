# www.tiziran.com

# Updated 22.Jan.2022
# This function use several methods for preprocessing and postprocessing images for better recognition calibration pattern.
# The pre build function of chessboard detection in OpenCV failed to recognize some calibration patterns

    # - cornerDetection.ipynb
        
    #     - It use several preprocessing and postprocessing steps to enhance corner detection use by camera calibration.

    #     - 3D multi camera calibration require detect and set points for all camera together  

    #     - if the calibration pattern images are not good, blur, ... it need to enhance it first then use corner points to detect and use for calibration process
# Farshid PirahanSiah
## Created 01.Jan.2022
## Last Update 08.Jan.2022

# The first step for camera calibration is corner detection. Based on my research, the calibration pattern image play important rule in the whole calibration process.

# 1. Camera calibration for multi-modal robot vision based on image quality assessment
# https://www.researchgate.net/profile/Farshid-Pirahansiah/publication/288174690_Camera_calibration_for_multi-modal_robot_vision_based_on_image_quality_assessment/links/5735bc2908aea45ee83c999e/Camera-calibration-for-multi-modal-robot-vision-based-on-image-quality-assessment.pdf 

# 2. Pattern image significance for camera calibration
# https://ieeexplore.ieee.org/abstract/document/8305440 

# 3. Camera Calibration and Video Stabilization Framework for Robot Localization
# https://link.springer.com/chapter/10.1007/978-3-030-74540-0_12 

import os
import sys
import cv2
import tqdm
import urllib
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray

from scipy import *
from scipy import signal as sig
from scipy.ndimage import gaussian_filter

sys.path.append(r'local_functions')
#from list_files import list_files
#from chessboard_corners import chessboard_corners

filePathURL = r"https://raw.githubusercontent.com/opencv/opencv/4.x/doc/tutorials/calib3d/camera_calibration/images/fileListImageUnDist.jpg" 
img_main = imread(filePathURL)
img_src= img_main.copy()
if len(img_main.shape)==3:
    gray_img=cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY) 
else:
    gray_img=img_src.copy()
#2
image_with_corners=gray_img.copy()
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
ret, corners = cv2.findChessboardCorners(gray_img, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray_img, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        image_with_corners = cv2.drawChessboardCorners(gray_img, CHECKERBOARD, corners2, ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_with_corners.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

