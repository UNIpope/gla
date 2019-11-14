# import the necessary packages:
import numpy as np
import cv2, easygui
from matplotlib import pyplot as plt
from matplotlib import image as image

def color_space(I):
    out = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    return out

def get_contours(I):
    

    color1 = np.array([109, 0, 141], dtype=np.uint8)   # white!
    color2 = np.array([139, 39, 255], dtype=np.uint8)   # less white/gray

    mask = cv2.inRange(img_hsv, color1, color2)
    #cont,_ =cv2.findContours(I, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)    

    cont, hierarchy = cv2.findContours(I, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(cont, key=cv2.contourArea, reverse=True)

    c = contours[0]
    print(c)

    I =cv2.drawContours(I,c,contourIdx=-1,color=(0,0,255),thickness=5)

    return I

def aline_im():
    pass


I = cv2.imread("MuirGlacier2004.jpg")

Img_bgr = color_space(I)
Img = get_contours(Img_bgr)

cv2.imshow("I", I)


cv2.imshow("out", Img)
key = cv2.waitKey(0)