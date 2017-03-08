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

# python scripts
import path as p


# load RGB image
def load_rgb_image(data_label_1,data_label_2):

    #Label preparation for directory operation
    if data_label_1 < 10 :
        data_label_1 = str(0)+str(data_label_1)
    else:
        data_label_1 = str(data_label_1)

    if data_label_2 < 10 :
        data_label_2 = str(0)+str(data_label_2)
    else:
        data_label_2 = str(data_label_2)

    path = p.data_path()+data_label_1+'/pcd'+data_label_1+data_label_2+'r.png'
    img = Image.open(path)
    img = img.crop((100,100,500,430))
    img.show()
    #img.save('rgb.png')


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
    for line in open(p.data_path()+data_label_1+'/pcd'+data_label_1+data_label_2+'cpos.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    #load image and draw rectangles
    image = Image.open(p.data_path()+data_label_1+'/dp'+data_label_1+data_label_2+'r.png')
    image = image.convert("RGB")
    image.show()
    #image.save('depth.png')
    draw = ImageDraw.Draw(image)

    shift_x = 100
    shift_y = 100
    scale = 1

    for i in range(len(xy_data)):

        x1 = (xy_data[i][0]-shift_x)*scale
        x2 = (xy_data[i][2]-shift_x)*scale
        x3 = (xy_data[i][4]-shift_x)*scale
        x4 = (xy_data[i][6]-shift_x)*scale

        y1 = (xy_data[i][1]-shift_y)*scale
        y2 = (xy_data[i][3]-shift_y)*scale
        y3 = (xy_data[i][5]-shift_y)*scale
        y4 = (xy_data[i][7]-shift_y)*scale

        draw.line((x1,y1)+(x2,y2), fill='yellow', width=2)
        draw.line((x2,y2)+(x3,y3), fill='green', width=2)
        draw.line((x3,y3)+(x4,y4), fill='yellow', width=2)
        draw.line((x4,y4)+(x1,y1), fill='green', width=2)

    #show image
    image.show()
    #image.save('depth_pos.png')

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
    for line in open(p.data_path()+data_label_1+'/pcd'+data_label_1+data_label_2+'cneg.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    #load image and draw rectangles
    image = Image.open(p.data_path()+data_label_1+'/dp'+data_label_1+data_label_2+'r.png')
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    shift_x = 100
    shift_y = 100
    scale = 1

    for i in range(len(xy_data)):

        x1 = (xy_data[i][0]-shift_x)*scale
        x2 = (xy_data[i][2]-shift_x)*scale
        x3 = (xy_data[i][4]-shift_x)*scale
        x4 = (xy_data[i][6]-shift_x)*scale

        y1 = (xy_data[i][1]-shift_y)*scale
        y2 = (xy_data[i][3]-shift_y)*scale
        y3 = (xy_data[i][5]-shift_y)*scale
        y4 = (xy_data[i][7]-shift_y)*scale

        draw.line((x1,y1)+(x2,y2), fill='red', width=2)
        draw.line((x2,y2)+(x3,y3), fill='blue', width=2)
        draw.line((x3,y3)+(x4,y4), fill='red', width=2)
        draw.line((x4,y4)+(x1,y1), fill='blue', width=2)

    #show image
    image.show()
    #image.save('depth_neg.png')

    return xy_data


#main
if __name__ == '__main__':

    #dlabel_1 = 1
    #dlabel_2 = 15

    d1 = 8
    d2 = 99

    dlabel_1 = np.random.randint(d1) + 1
    dlabel_2 = np.random.randint(d2) + 1

    print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)
    load_rgb_image(dlabel_1,dlabel_2)
    positive_vartices = load_positive_dataset(dlabel_1,dlabel_2)
    negative_vartices = load_negative_dataset(dlabel_1,dlabel_2)
