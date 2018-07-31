import cv2
import numpy as np
import time

def getLines(image_name):
    ## Convert from RGB to gray
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## Convert to binary image
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    
    
    ## Calculate rotated image
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    
    (cx,cy), (w,h), ang = ret
    if w>h:
        w,h = h,w
        ang += 90
    
    M = cv2.getRotationMatrix2D((cx,cy), 0, 1.0)
    rotated = threshed
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
    #cv2.imshow("", rotated)
    #cv2.waitKey()
    
    ## Draw upper and lower lines for each text line
    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 10
    H,W = img.shape[:2]
    uppers = [y-5 for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y+5 for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    print(len(uppers))
    print(len(lowers))
    
    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    minDistance = 5
    count = 0
    croppedImages = []
    for i in range(len(uppers)):
        print(uppers[i])
        print(lowers[i])
        tooClose = False
        if abs(uppers[i] - lowers[i]) < minDistance:
            tooClose = True
            print("too close")
        if not tooClose:
            croppedImages.append(rotated[uppers[i]:lowers[i],:])
            #cv2.line(rotated, (0,uppers[i]), (W, uppers[i]), (255,0,0), 1)
            #cv2.line(rotated, (0,lowers[i]), (W, lowers[i]), (0,255,0), 1)
        
    return np.array(croppedImages)
    #cv2.imwrite("result.png", croppedImages[0])

getLines("index.jpg")