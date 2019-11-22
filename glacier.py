import numpy as np
import cv2
import math
from tkinter import *
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

newImagef = None
oldImagef = None

def selectImage():
    f = easygui.fileopenbox()

    return f

def closeMask(mask,x,y):
    shapeC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(x,y))
    Cmask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,shapeC)
    return Cmask

def openMask(mask,x,y):
    shapeO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(x,y))
    Omask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,shapeO)
    return Omask

def JoinOnMask(foreground,background,mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    maskInv = ~mask
    foregroundROI = cv2.bitwise_and(foreground,foreground,mask=mask)
    backgroundROI = cv2.bitwise_and(background,background,mask=maskInv)
    merged = foregroundROI + backgroundROI
    return merged

def draw(event, x, y, flags, param):
    global start, drawing, userMask, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            end = (x,y)
            cv2.line(userMask, start, end, (255, 255, 255), 40)
            img = JoinOnMask(glacierMerged, img, userMask)
            start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def im1Select():
    oldImageSelect = selectImage()

    global oldImagef
    oldImagef = oldImageSelect



def im2Select():
    newImageSelect = selectImage()

    global newImagef
    newImagef = newImageSelect


window = Tk() 
window.geometry("500x500")
window.title("Repeat Photography")

label1=Label(window, text="Quantify Glacial Erosion \n in two photos", relief="solid", width=20, font=("arial",19,"bold"))
label1.place(x=90,y=53)

im1Button = Button(window, text= "Old Image", width= 12, command=im1Select, bg="brown", fg="white")
im1Button.place(x=130,y=360)

im2Button = Button(window, text= "New Image", width= 12, command=im2Select, bg="brown", fg="white")
im2Button.place(x=260,y=360)

GoButton = Button(window, text="GO",width = 12, command=window.destroy)
GoButton.place(x=200, y=280)

 
window.mainloop() 

oldImage = cv2.imread(oldImagef)
newImage = cv2.imread(newImagef)


glacierMask = oldImage.copy()
glacierMaskBorder = oldImage.copy()
glacierMask[:,:,:] = 0
glacierMaskBorder[:,:,:] = 0

lower = np.array([0,0,0])
upper = np.array([140,140,140])
mask = cv2.inRange(newImage,lower,upper)

closed = closeMask(mask,20,30)
opened = openMask(closed,15,15)

mountain = cv2.bitwise_and(oldImage,oldImage,mask=opened)

lower = np.array([170,150,150])
upper = np.array([255,255,255])
whiteMask = cv2.inRange(mountain,lower,upper)

contours, hierarchy = cv2.findContours(whiteMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = 0
i = 0
for contour in contours:
    if cv2.contourArea(contour) > area:
        area = cv2.contourArea(contour)
        contourI = i
    i += 1

cv2.drawContours(glacierMask,contours,contourI,(255,255,255),-1)
glacierMask = closeMask(glacierMask,15,15)
oldGlacierNew = JoinOnMask(oldImage,newImage,glacierMask)

glacierMask = cv2.cvtColor(glacierMask, cv2.COLOR_BGR2GRAY)
glacierContour, hierarchy = cv2.findContours(glacierMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(glacierMaskBorder,glacierContour,0,(255,255,255),2)
glacierMaskBorder = cv2.cvtColor(glacierMaskBorder, cv2.COLOR_BGR2GRAY)
glacierMerged = cv2.inpaint(oldGlacierNew,glacierMaskBorder, 1, cv2.INPAINT_TELEA)
glacierMerged = cv2.cvtColor(glacierMerged, cv2.COLOR_BGR2RGB)


windowName = 'Drawing'
drawing = False
userMask = newImage.copy()
userMask[:,:,:] = 0

img = newImage.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, draw)
while (True):
    cv2.imshow(windowName, img)
    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()

plt.subplot(2, 2, 1)
plt.imshow(oldImage)
plt.subplot(2, 2, 3)
plt.imshow(newImage)
plt.subplot(2, 2, 2)
plt.imshow(whiteMask)
plt.subplot(2, 2, 4)
plt.imshow(mountain)
plt.show()
