# python library
import numpy as np
from numpy.random import *

# OpenCV
import cv2


# load picture data
def find_object(path):

    #1. read image
    img = cv2.imread(path)
    #print img.shape

    #2. gray image
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #3. blur image
    g_blur = cv2.GaussianBlur(grayed,(21,21),0)

    #4. binary image
    under_thresh = 180
    max_value = 255

    _, binary = cv2.threshold(g_blur, under_thresh, max_value, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    #5. recognize contour and rectangle
    contour, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = np.copy(img)

    # area threshold
    min_area = 100
    #max_area = 17000
    max_area = 20000
    #max_area = 15000

    object_contour = [cnt for cnt in contour if cv2.contourArea(cnt) < max_area and cv2.contourArea(cnt) > min_area]
    cv2.drawContours(img_contour, object_contour, -1, (255,0,255),2)

    object_rec = []

    for i in range(len(object_contour)):
        object_rec.append(cv2.boundingRect(object_contour[i]))

    if len(object_rec)  == 0:
        print "\n error: could not find objects. \n"
    # else:
    #     print "\n amount of objects: "+str(len(object_rec)) +"\n"

    return np.array(object_contour),np.array(object_rec)
