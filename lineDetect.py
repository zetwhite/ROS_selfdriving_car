import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob

class Line : 
    def __init__(self, start, end) : 
        self.start = start
        self.end = end
    def update(self, start, end) : 
        self.start = start 
        self.end = end
    def __str__(self) : 
        if(self.start == 0 and self.end == 0) : 
            return "not setted"
        return "(%d, %d), min = %d" % (self.start, self.end, (self.start + self.end) // 2) 


cap = cv2.VideoCapture('./track-s.mkv')
width = 0
height = 0
fig, ax = plt.subplots(2, 2)

lineSetted = False 
lines= [Line(0, 0) , Line(0, 0), Line(0, 0)] 
#leftLine = Line(0, 0)
#midLine = Line(0, 0)
#rightLine = Line(0, 0)


def init() : 
    global cap, width, height, fig, ax, lineSetted
    cap = cv2.VideoCapture('./track-s.mkv')
    ret, frame = cap.read() 
    height, width = frame.shape[:2]
    plt.ion()

def canny(image) : 
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image): 
    region = np.array([[(0, height - 30), (width, height - 30), (width - 200, 250), (200, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def perspection(image):
    pts1 = np.float32([[width//5, height//2], [width//5 * 4, height//2], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspection_image = cv2.warpPerspective(image, M, (width, height))
    return perspection_image

def get_histogram(image):
    cuttedImage = image[height//4:, :]
    ret, image_bin = cv2.threshold(cuttedImage, 125, 255, cv2.THRESH_BINARY)
    hist = np.sum(image_bin, axis = 0)
    return hist

def imageShow(img1, img2, img3, img4) : 
    ax[0, 0].imshow(img1, cmap='gray', vmin = 0, vmax = 255)
    ax[0, 1].imshow(img2, cmap='gray', vmin = 0, vmax = 255)
    ax[1, 0].imshow(img3, cmap='gray', vmin = 0, vmax = 255)
    ax[1, 1].plot(img4, color = 'b')
    plt.pause(0.00001)
    ax[1, 1].lines.pop(0)
    plt.show(False)
    plt.draw()

def getLine(histogram) : 
    global lineSetted, lines
    start = []
    end = [] 
    thres = 100
    prev = histogram[0] 
    for i in range(1, len(histogram)) : 
        now = histogram[i]
        if prev < thres and now > thres : 
            start.append(i) 
        if prev > thres and now < thres : 
            end.append(i)
        prev = now 
    #print("start : ", start)
    #print("end : " , end)
    
    if len(start) != len(end) : 
        return
    if lineSetted == False and len(start) == 3 : 
        lineSetted = True
        print("At First, Lines are setted!!!")
        for i in range(3) : 
            lines[i].update(start[i], end[i]) 
        return 


    #do bipartite matching!!!
    if lineSetted == True and len(start) == 1 : 
        minIndex = 0 
        mindiff = 10000
        for i in range(3) : 
            diff = abs(lines.start - start[i]) + abs(lines.end - end[i]) 
            if( diff < mindiff) : 
                mindiff = diff 
                minIndex = i 
        lines[i].update(start[0], end[0])
        return 

    if lineSetted == True and len(start) == 2 : 
        minIndex = set()
        mindiff = 10000 
        for i in range(3) : 
            dif = 0
            idx = 0
            for j in set(range(3)) - set([i]) : 
                dif = abs(lines[j].start - start[idx]) + abs(lines[j].end - end[idx])
                idx = idx + 1 
            if dif < mindiff : 
                mindiff = dif 
                minIndex = set(range(3)) - set([i])
        lines[list(minIndex)[0]].update(start[0], end[0])
        lines[list(minIndex)[1]].update(start[1], end[1]) 
        return 

    if lineSetted == True and len(start) == 3 : 
        for i in range(3) : 
            lines[i].update(start[i], end[i]) 
        return 


init()
while (cap.isOpened) : 
    ret, frame = cap.read()
    
    if ret : 
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", grayframe)
        cannyed_image = canny(grayframe) 
        cropped_image = region_of_interest(cannyed_image)
        topview_image = perspection(cropped_image)
        hist = get_histogram(topview_image) 
        getLine(hist)
        if lineSetted : 
            print("==line==")
            for i in range(3) : 
                print(lines[i]) 
        imageShow(cannyed_image, cropped_image, topview_image, hist)
        k = cv2.waitKey(1)
        if k == 27 : 
            break
    else : 
            break
#plt.show() 
cap.release() 
cv2.destroyAllWindows()
