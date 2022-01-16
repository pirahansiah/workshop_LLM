import numpy as np
import cv2

def chessboard_corners(img):
    '''
    search and find points in calibration pattern image 
    
    input:
        img: the calibration pattern "chessboard" image file
        
    '''
    max_a=4
    max_b=4
    found_pints=False
    for a in range(12,35):
        for b in range (12,35):                
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
            objp = np.zeros((36*37,3), np.float32)
            objp[:,:2] = np.mgrid[0:37,0:36].T.reshape(-1,2)
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            gray=img.copy()
            ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)                                
                imgpoints.append(corners)                
                if (max_a<a and max_b<b):
                    max_a=a
                    max_b=b                           
                    found_pints=True                                                                      
                print(f"a= {a},b = {b} result {ret}  "
                    f'  found  '
                    )                
            else:
                #print(f"a= {a},b = {b} result {ret}")    
                pass
    if (found_pints):
        cv2.drawChessboardCorners(img, (a,b), corners2, ret)                    
        return img
    else:
        return False