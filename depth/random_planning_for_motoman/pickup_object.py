# python library
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt

# OpenCV
import cv2

# python scripts
import path as p


# label preparation
def label_handling(data_label_1,data_label_2):

    data_label = []

    if data_label_1 < 10 :
        data_label.append(str(0)+str(data_label_1))
    else:
        data_label.append(str(data_label_1))

    if data_label_2 < 10 :
        data_label.append(str(0)+str(data_label_2))
    else:
        data_label.append(str(data_label_2))

    return data_label


# load picture data
def find_object_from_RGB(data_label):

    #1. read image
    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
    img = cv2.imread(path)

    #2. gray image
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #3. blur image
    # blur parameter
    k_size = 21
    g_blur = cv2.GaussianBlur(grayed,(k_size,k_size),0)

    #4. binary image
    # binary parameters
    under_thresh = 180
    max_value = 255

    _, binary = cv2.threshold(g_blur, under_thresh, max_value, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    #5. recognize contour and rectangle
    contour, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = np.copy(img)

    # area threshold
    min_area = 100
    max_area = 15000

    object_contour = [cnt for cnt in contour if cv2.contourArea(cnt) < max_area and cv2.contourArea(cnt) > min_area]
    cv2.drawContours(img_contour, object_contour, -1, (255,0,255),2)

    object_rec = []

    for i in range(len(object_contour)):
        object_rec.append(cv2.boundingRect(object_contour[i]))
        print 'x:'+str(object_rec[i][0])+' y:'+str(object_rec[i][1])+' w:'+str(object_rec[i][2])+' h:'+str(object_rec[i][3])
        cv2.rectangle(img_contour, (object_rec[i][0], object_rec[i][1]), (object_rec[i][0] + object_rec[i][2], object_rec[i][1] + object_rec[i][3]), (255, 100, 100), 2)

    if len(object_rec)  == 0:
        print "error: could not find objects."
    else:
        print "amount of rectangles: "+str(len(object_rec))

    plt.figure(0)
    plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    cv2.imwrite('pictures/rgb.png',img_contour)

    return np.array(object_contour),np.array(object_rec)
