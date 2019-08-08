"""'''# Farshid Pirahansiah 8 August 2019'''"""
'''# Farshid Pirahansiah 8 August 2019'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
class image_processing_class:
    def __init__(self):
        pass
    def show_image_plt(self,image):    
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.title('ORIGINAL')
        plt.show()  
    def show_image_opencv(self,image):
        cv2.imshow("farshid",image)
        cv2.waitKey(1000)
    def save_image_opencv(self,filename,img):
        if len(filename)==0:
            cv2.imwrite("farshid.jpg", img)
        else:
            cv2.imwrite([filename], [img])    
    def cartoon_image(self,image):
        num_down = 5#2 # number of downsampling steps
        num_bilateral = 9#7 # number of bilateral filtering steps
        img_rgb = image
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        # upsample image to original size
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
        #STEP 2 & 3
        #Use median filter to reduce noise
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        #STEP 4
        #Use adaptive thresholding to create an edge mask
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,    cv2.ADAPTIVE_THRESH_MEAN_C,    cv2.THRESH_BINARY,    blockSize=9,    C=2)
        # Step 5
        # Combine color image with edge mask & display picture
        # convert back to color, bit-AND with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_cartoon = cv2.bitwise_and(img_color, img_edge)
        # display
        image=img_cartoon
        cv2.imshow("cartoon", img_cartoon)
        cv2.waitKey(0)
        return image
    def face_detection_webcam(self):
        cap=cv2.VideoCapture(0)
        while(cap.isOpened()):
            _,frame=cap.read()     
            rows,cols,channels = frame.shape        
            img=frame.copy()
            face_cascade = cv2.CascadeClassifier('farshid/haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('farshid/haarcascades/haarcascade_eye.xml')    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # display
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,10)
            fontScale              = 0.3
            fontColor              = (255,0,0)
            lineType               = 2
            cv2.putText(frame,'original', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(img,'face detection', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            img_concat0 =np.concatenate((frame, img), axis=1)
            cv2.imshow('Farshid PirahanSiah; Computer Vision Workshop', img_concat0 )
            key=cv2.waitKey(1)
            if (key == ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()
            
           
    def face_detection(self,img):
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def thresholding(self):
        cap=cv2.VideoCapture(0)
        size_image=300
        while(cap.isOpened()):
            _,frame=cap.read()     
            rows,cols,channels = frame.shape
            frame_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            ret, th1 = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY)            
            ret, th2 = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_OTSU)
            ret, th3 = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_TOZERO_INV)
            th4 = cv2.adaptiveThreshold(frame_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
            th5 = cv2.adaptiveThreshold(frame_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)                        
            frame = cv2.resize(frame,(int(size_image),int(size_image)))
            frame_gray = cv2.resize(frame_gray,(int(size_image),int(size_image)))
            th1 = cv2.resize(th1,(int(size_image),int(size_image)))
            th2 = cv2.resize(th2,(int(size_image),int(size_image)))
            th3 = cv2.resize(th3,(int(size_image),int(size_image)))
            th4 = cv2.resize(th4,(int(size_image),int(size_image)))
            th5 = cv2.resize(th5,(int(size_image),int(size_image)))            
            frame_gray=cv2.cvtColor(frame_gray,cv2.COLOR_GRAY2BGR)
            th1=cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)
            th2=cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
            th3=cv2.cvtColor(th3,cv2.COLOR_GRAY2BGR)
            th4=cv2.cvtColor(th4,cv2.COLOR_GRAY2BGR)
            th5=cv2.cvtColor(th5,cv2.COLOR_GRAY2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,10)
            fontScale              = 0.3
            fontColor              = (255,0,0)
            lineType               = 2
            cv2.putText(th1,'THRESH_BINARY', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(th2,'THRESH_OTSU', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(th3,'THRESH_TOZERO_INV', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(th4,'ADAPTIVE_THRESH_MEAN_C', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(th5,'ADAPTIVE_THRESH_GAUSSIAN_C', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(frame,'Original', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            img_concat0 =cv2.vconcat([frame, frame_gray])
            img_concat1 =cv2.vconcat([th1, th2])
            img_concat2 =cv2.vconcat([th5, th4])
            img_concat_all =cv2.hconcat([img_concat0,img_concat1,img_concat2])
            cv2.imshow('Farshid PirahanSiah; Computer Vision Workshop', img_concat_all )
            key=cv2.waitKey(1)
            if (key == ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()
    def color(self):
        size_image=300
        cap=cv2.VideoCapture(0)
        while(cap.isOpened()):
            _,frame=cap.read()     
            rows,cols,channels = frame.shape        
            original = cv2.resize(frame,(int(size_image),int(size_image)))        
            f1=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)               
            f2=cv2.cvtColor(original,cv2.COLOR_RGB2HSV)
            f3=cv2.cvtColor(original,cv2.COLOR_RGB2YUV)
            f4=cv2.cvtColor(original,cv2.COLOR_RGB2YCrCb)
            f5=cv2.cvtColor(original,cv2.COLOR_RGB2HLS)
            f6=cv2.cvtColor(original,cv2.COLOR_RGB2Luv)
            f7=cv2.cvtColor(original,cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,10)
            fontScale              = 0.3
            fontColor              = (255,0,0)
            lineType               = 2
            cv2.putText(f1,'COLOR_RGB2GRAY', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(f2,'COLOR_RGB2HSV', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(f3,'COLOR_RGB2YUV', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(f4,'COLOR_RGB2YCrCb', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(f5,'COLOR_RGB2HLS', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(f6,'COLOR_RGB2Luv', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(f7,'COLOR_RGB2RGBA', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(original,'Original', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            f1=cv2.cvtColor(f1,cv2.COLOR_GRAY2RGB) 
            img_concat0 =np.concatenate((original, f1), axis=0)
            img_concat1 =np.concatenate((f2, f3), axis=0)
            img_concat2 =np.concatenate((f4, f5), axis=0) 
            img_concat3 =np.concatenate((f6, f7), axis=0) 
            img_concat_all = np.concatenate((img_concat0,img_concat1,img_concat2,img_concat3 ), axis=1)

            cv2.imshow('Farshid PirahanSiah; Computer Vision Workshop', img_concat_all )
            key=cv2.waitKey(1)
            if (key == ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()
# cartoon
    def cartoon_webcam(self):
        cap=cv2.VideoCapture(0)
        while(cap.isOpened()):
            _,frame=cap.read()     
            rows,cols,channels = frame.shape     
            image=frame
            num_down = 5#2 # number of downsampling steps
            num_bilateral = 9#7 # number of bilateral filtering steps
            img_rgb = image
            # downsample image using Gaussian pyramid
            img_color = img_rgb
            for _ in range(num_down):
                img_color = cv2.pyrDown(img_color)
            # repeatedly apply small bilateral filter instead of
            # applying one large filter
            for _ in range(num_bilateral):
                img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
            # upsample image to original size
            for _ in range(num_down):
                img_color = cv2.pyrUp(img_color)
            #STEP 2 & 3
            #Use median filter to reduce noise
            # convert to grayscale and apply median blur
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)
            #STEP 4
            #Use adaptive thresholding to create an edge mask
            # detect and enhance edges
            img_edge = cv2.adaptiveThreshold(img_blur, 255,    cv2.ADAPTIVE_THRESH_MEAN_C,    cv2.THRESH_BINARY,    blockSize=9,    C=2)
            # Step 5
            # Combine color image with edge mask & display picture
            # convert back to color, bit-AND with color image
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            img_cartoon = cv2.bitwise_and(img_color, img_edge)
            # display
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,10)
            fontScale              = 0.3
            fontColor              = (255,0,0)
            lineType               = 2
            cv2.putText(frame,'original', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(img_cartoon,'cartoon', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            img_concat0 =np.concatenate((frame, img_cartoon), axis=1)
            cv2.imshow('Farshid PirahanSiah; Computer Vision Workshop', img_concat0 )
            key=cv2.waitKey(1)
            if (key == ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()




    def mainfarshid(self):        
        print("farshid pirahansiah")
        print(cv2.__version__)
        cap=cv2.VideoCapture(0)
        if(cap.isOpened()):
            _,frame=cap.read()     
            rows,cols,channels = frame.shape
            cv2.imshow('farshid original',frame)
            cv2.waitKey(1000)
            #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            #ret, mask = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
            #ret, mask = cv2.threshold(frame, 127, 255, cv2.THRESH_OTSU)
            th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
            th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
            show_image_plt(th2)
            show_image_opencv(th3)
            #fp.show_image_opencv(th3)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            cap.release()
            cv2.destroyAllWindows()




if __name__== "__main__":
    fp=Farshid()
    fp.color()
    # thresholding()
    