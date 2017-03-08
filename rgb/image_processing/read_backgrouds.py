# python library
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt

# OpenCV
import cv2

# python scripts
import path as p


# label preparation
def background_label():

    label = []

    for i in range(2,9):
        label.append(str(0)+str(i))
    for i in range(10,14):
        label.append(str(i))

    return label


# read background images
def background():

    data_label = background_label()
    min_sum = 0

    for i in range(len(data_label)):
        img = 0
        path = p.data_path()+'backgrounds/pcdb00'+data_label[i]+'r.png'
        img = cv2.imread(path)

        #print img.shape
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_blur = cv2.GaussianBlur(grayed,(21,21),0)
        min_sum += g_blur[200:480,100:500].min()

        #plt.figure(i)
        #plt.imshow(cv2.cvtColor(g_blur[200:480,100:500], cv2.COLOR_GRAY2RGB))
        #plt.axis('off')

    average = min_sum/len(data_label)

    return average


if __name__ == '__main__':

    average = background()

    print average
    plt.show()
