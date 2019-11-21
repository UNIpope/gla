import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

def close(mask,x,y):
    shapeC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(x,y))
    Cmask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,shapeC)
    return Cmask

def open(mask,x,y):
    shapeO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(x,y))
    Omask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,shapeO)
    return Omask

f = "Glacier.jpg"
I = cv2.imread(f)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
h,w,c = I.shape
splitHeight = int(h/2)

oldImage = I[0:splitHeight,:,:]
newImage = I[splitHeight:h,:,:]

alpha = 0.5
overlayed = cv2.addWeighted(newImage, alpha, oldImage, 1 - alpha,0, 0)

lower = np.array([50,50,80])
upper = np.array([140,140,140])
mask = cv2.inRange(newImage,lower,upper)

closed = close(mask,20,30)
opened = open(closed,15,15)

mountain = cv2.bitwise_and(oldImage,oldImage,mask=opened)

lower = np.array([190,190,190])
upper = np.array([255,255,255])
glacierMask = cv2.inRange(mountain,lower,upper)

closedGM = close(glacierMask,5,5)

contours, hierarchy = cv2.findContours(glacierMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = 0
for contour in contours:
    if cv2.contourArea(contour) > area:
        hull= cv2.convexHull(contour)
        area = cv2.contourArea(contour)

cv2.drawContours(mountain,[hull],0,(0,0,255),2)

plt.subplot(2, 2, 1)
plt.imshow(oldImage)
plt.subplot(2, 2, 3)
plt.imshow(newImage)
plt.subplot(2, 2, 2)
plt.imshow(mountain)
plt.subplot(2, 2, 4)
plt.imshow(closedGM)
plt.show()
