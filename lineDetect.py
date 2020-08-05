import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob

cap = cv2.VideoCapture('./track-s.mkv')

def canny(image) : 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image): 
    height, width = image.shape[:2] 
    region = np.array([[(0, height - 30), (width, height - 30), (width - 200, 250), (200, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def perspection(image):
    height, width = image.shape[:2]
    pts1 = np.float32([[width/5, height/2], [width/5 * 4, height/2], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspection_image = cv2.warpPerspective(image, M, (width, height))
    return perspection_image

while (cap.isOpened) : 
    ret, frame = cap.read()
    
    if ret : 
        #cv2.imshow("video", frame)
        cannyed_image = canny(frame) 
        cv2.imshow("cannyed", cannyed_image)
        cropped_image = region_of_interest(cannyed_image)
        cv2.imshow("cropped", cropped_image)
        topview_image = perspection(cropped_image)
        cv2.imshow("topview", topview_image)
        
        k = cv2.waitKey(33)
        if k == 27 : 
            break
    else : 
            break
        
cap.release() 
cv2.destroyAllWindows()
