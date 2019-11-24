# Step 1
# Create two copies of the original image, and zero them out, this will be used later for masks,
# but the images need to be the same size as the original images
# Step2
# Create a lower and upper range to specify the grey background in the new image,
# create a mask from this range
# Step3
# Close the mask and then open it
# Mask the old image using the mountain ROI from the new image
# Step 4
# Specify a range of white to create a mask that is areas of white that were present in the
# old image where there is mountain in the new image
# Step 5
# Create contours from the white mask. Find the largest white contour by iterating through
# every contour and comparing their area and marking the largest area contour location
# Step 6
# Create a mask of the largest white contour by drawing it in white on one of the blank
# images created earlier, close the mask to remove false negative.
# Merge the old glacier on to the new image
# Step 7
# Blend the edges of the glacier using the inpaint function. The mask is created from the outline
# of the largest contour earlier
# Step 8
# Set variables for the drawing function and call the setMouseCallback, and imshow in a while loop
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

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

# Step 0
f = "glacier.jpg"
I = cv2.imread(f)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
h,w,c = I.shape
splitHeight = int(h/2)
oldImage = I[0:splitHeight,:,:]
newImage = I[splitHeight:(2*splitHeight),:,:]

# f = easygui.fileopenbox()
# newImage = cv2.imread(f)
#
# f2 = easygui.fileopenbox()
# oldImage = cv2.imread(f2)

# Step 1
glacierMask = oldImage.copy()
glacierMaskBorder = oldImage.copy()
glacierMask[:,:,:] = 0
glacierMaskBorder[:,:,:] = 0

# Step 2
lower = np.array([0,0,0])
upper = np.array([140,140,140])
greyMask = cv2.inRange(newImage,lower,upper)

# Step 3
greyMask = closeMask(greyMask,20,30)
greyMask = openMask(greyMask,15,15)
mountain = cv2.bitwise_and(oldImage,oldImage,mask=greyMask)

# Step 4
lower = np.array([170,150,150])
upper = np.array([255,255,255])
whiteMask = cv2.inRange(mountain,lower,upper)

# Step 5
contours, hierarchy = cv2.findContours(whiteMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = 0
i = 0
for contour in contours:
    if cv2.contourArea(contour) > area:
        area = cv2.contourArea(contour)
        contourI = i
    i += 1

# Step 6
cv2.drawContours(glacierMask,contours,contourI,(255,255,255),-1)
glacierMask = closeMask(glacierMask,15,15)
oldGlacierNew = JoinOnMask(oldImage,newImage,glacierMask)

# Step 7
glacierMask = cv2.cvtColor(glacierMask, cv2.COLOR_BGR2GRAY)
glacierContour, hierarchy = cv2.findContours(glacierMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(glacierMaskBorder,glacierContour,0,(255,255,255),2)
glacierMaskBorder = cv2.cvtColor(glacierMaskBorder, cv2.COLOR_BGR2GRAY)
glacierMerged = cv2.inpaint(oldGlacierNew,glacierMaskBorder, 1, cv2.INPAINT_TELEA)
glacierMerged = cv2.cvtColor(glacierMerged, cv2.COLOR_BGR2RGB)

# Step 8
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
