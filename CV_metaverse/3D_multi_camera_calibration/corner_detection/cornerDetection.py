# Updated 07.Jan.2022
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
        
print("farshid")
import cv2
