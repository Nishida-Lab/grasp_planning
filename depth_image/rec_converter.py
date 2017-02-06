#python library
import numpy as np
from matplotlib import pyplot as plt
import os

# OpenCV
import cv2

#python library
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#PCL
import pcl


#load dataset and draw grasp rectangle
def load_positive_dataset(data_label_1,data_label_2):

    #Label preparation for directory operation
    if data_label_1 < 10 :
        data_label_1 = str(0)+str(data_label_1)
    else:
        data_label_1 = str(data_label_1)

    if data_label_2 < 10 :
        data_label_2 = str(0)+str(data_label_2)
    else:
        data_label_2 = str(data_label_2)

    #load grasping rectangles
    xy_data = []
    #for line in open('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'cpos.txt').readlines():
    for line in open('test_data/'+'pcd'+data_label_1+data_label_2+'cpos.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    #load image and draw rectangles
    scaled = (360,270)
    image = Image.open('test_data/'+'dp'+data_label_1+data_label_2+'r.png')
    image = image.convert("RGB")
    image = image.resize(scaled)
    image.show()
    draw = ImageDraw.Draw(image)

    rotation = np.zeros((3,3))
    theta = -6*(np.pi/180)
    rotation[0][0] = np.cos(theta)
    rotation[0][1] = -np.sin(theta)
    rotation[0][2] = scaled[0]/2-scaled[0]/2*np.cos(theta)+scaled[1]/2*np.sin(theta)
    rotation[1][0] = np.sin(theta)
    rotation[1][1] = np.cos(theta)
    rotation[1][2] = scaled[1]/2-scaled[0]/2*np.sin(theta)-scaled[1]/2*np.cos(theta)
    rotation[2][2] = 1

    shift_x = 55
    shift_y = 140

    for i in range(len(xy_data)):

        x1 = xy_data[i][0]-shift_x
        x2 = xy_data[i][2]-shift_x
        x3 = xy_data[i][4]-shift_x
        x4 = xy_data[i][6]-shift_x

        y1 = xy_data[i][1]-shift_y
        y2 = xy_data[i][3]-shift_y
        y3 = xy_data[i][5]-shift_y
        y4 = xy_data[i][7]-shift_y

        p1 = [[x1],[y1],[1]]
        p2 = [[x2],[y2],[1]]
        p3 = [[x3],[y3],[1]]
        p4 = [[x4],[y4],[1]]

        pd1 = rotation.dot(p1)
        pd2 = rotation.dot(p2)
        pd3 = rotation.dot(p3)
        pd4 = rotation.dot(p4)

        draw.line((pd1[0],pd1[1])+(pd2[0],pd2[1]), fill='yellow', width=2)
        draw.line((pd2[0],pd2[1])+(pd3[0],pd3[1]), fill='green', width=2)
        draw.line((pd3[0],pd3[1])+(pd4[0],pd4[1]), fill='yellow', width=2)
        draw.line((pd4[0],pd4[1])+(pd1[0],pd1[1]), fill='green', width=2)

    #show image
    image.show()

    return xy_data


#load dataset and draw grasp rectangle
def load_negative_dataset(data_label_1,data_label_2):

    #Label preparation for directory operation
    if data_label_1 < 10 :
        data_label_1 = str(0)+str(data_label_1)
    else:
        data_label_1 = str(data_label_1)

    if data_label_2 < 10 :
        data_label_2 = str(0)+str(data_label_2)
    else:
        data_label_2 = str(data_label_2)

    #load grasping rectangles
    xy_data = []
    #for line in open('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'cpos.txt').readlines():
    for line in open('test_data/'+'pcd'+data_label_1+data_label_2+'cneg.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    #load image and draw rectangles
    scaled = (360,270)
    image = Image.open('test_data/'+'dp'+data_label_1+data_label_2+'r.png')
    image = image.convert("RGB")
    image = image.resize(scaled)
    draw = ImageDraw.Draw(image)

    rotation = np.zeros((3,3))
    theta = -6*(np.pi/180)
    rotation[0][0] = np.cos(theta)
    rotation[0][1] = -np.sin(theta)
    rotation[0][2] = scaled[0]/2-scaled[0]/2*np.cos(theta)+scaled[1]/2*np.sin(theta)
    rotation[1][0] = np.sin(theta)
    rotation[1][1] = np.cos(theta)
    rotation[1][2] = scaled[1]/2-scaled[0]/2*np.sin(theta)-scaled[1]/2*np.cos(theta)
    rotation[2][2] = 1

    shift_x = 55
    shift_y = 140

    for i in range(len(xy_data)):

        x1 = xy_data[i][0]-shift_x
        x2 = xy_data[i][2]-shift_x
        x3 = xy_data[i][4]-shift_x
        x4 = xy_data[i][6]-shift_x

        y1 = xy_data[i][1]-shift_y
        y2 = xy_data[i][3]-shift_y
        y3 = xy_data[i][5]-shift_y
        y4 = xy_data[i][7]-shift_y

        p1 = [[x1],[y1],[1]]
        p2 = [[x2],[y2],[1]]
        p3 = [[x3],[y3],[1]]
        p4 = [[x4],[y4],[1]]

        pd1 = rotation.dot(p1)
        pd2 = rotation.dot(p2)
        pd3 = rotation.dot(p3)
        pd4 = rotation.dot(p4)

        draw.line((pd1[0],pd1[1])+(pd2[0],pd2[1]), fill='red', width=2)
        draw.line((pd2[0],pd2[1])+(pd3[0],pd3[1]), fill='blue', width=2)
        draw.line((pd3[0],pd3[1])+(pd4[0],pd4[1]), fill='red', width=2)
        draw.line((pd4[0],pd4[1])+(pd1[0],pd1[1]), fill='blue', width=2)

    #show image
    image.show()

    return xy_data


#main
if __name__ == '__main__':

    #demo
    dlabel_1 = 1
    dlabel_2 = 16

    #dlabel_2 = 40

    print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)
    positive_vartices = load_positive_dataset(dlabel_1,dlabel_2)
    negative_vartices = load_negative_dataset(dlabel_1,dlabel_2)
