# -*- coding: utf-8 -*-

#python library
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
import sys
from scipy import misc
import shutil
import os
import random
import time

# OpenCV
import cv2

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers

#python script
import network_structure as nn
import pickup_object as po
import random_rectangle as rr
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


# load the depth image data
def load_depth_image(path,scale):

    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(grayed,(img.shape[0]/scale,img.shape[1]/scale))
    img_array = np.asanyarray(resized_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(resized_img.size,1))
    img_list = []
    for i in range(len(img_array)):
        img_list.append(img_array[i][0]/255.0)

    return img,img_list


# generate input data for CNN
def generate_input_data(path,rec_list,scale):
    img,img_list = load_depth_image(path,scale)
    x = rec_list + img_list
    x = np.array(x,dtype=np.float32).reshape((1,33008))
    return x,img


# calculate z
def calculate_z(path,rec,scale):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grayed)
    x1 = int(rec[1]*scale)
    y1 = int(rec[0]*scale)
    x2 = int(rec[5]*scale)
    y2 = int(rec[4]*scale)
    ml = np.max(img_array[np.min([x1,x2]):np.max([x1,x2]),np.min([y1,y2]):np.max([y1,y2])])
    z =round(ml*(150/255.0),2)
    return z


# draw the object area
def draw_object_area(img,object_area,rec_area):

    for i in range(len(rec_area)):

        for j in range(len(object_area[i])):
            for k in range(2):
                object_area[i][j][0][k] = object_area[i][j][0][k] - 100

        cv2.rectangle(img, (rec_area[i][0]-100, rec_area[i][1]-100), (rec_area[i][0]+rec_area[i][2]-100, rec_area[i][1]+rec_area[i][3]-100), (255,0,0), 1)
    cv2.drawContours(img, object_area, -1, (255,0,255),1)


# draw the grasp rectangle
def draw_grasp_rectangle(img,rec,scale):
    color = [(0,255,255), (0,255,0)]
    rec = (np.array(rec)*scale).astype(np.int32)
    cv2.line(img,(rec[0],rec[1]),(rec[2],rec[3]), color[0], 2)
    cv2.line(img,(rec[2],rec[3]),(rec[4],rec[5]), color[1], 2)
    cv2.line(img,(rec[4],rec[5]),(rec[6],rec[7]), color[0], 2)
    cv2.line(img,(rec[6],rec[7]),(rec[0],rec[1]), color[1], 2)
    plt.figure(1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    cv2.imwrite('pictures/depth.png',img)


#main
if __name__ == '__main__':

    directory_n = input('Directory No > ')
    picture_n = input('Image No > ')

    # random checking
    #directory_n = randint(9)+1
    #picture_n = randint(40)+1

    scale = 2

    print 'directory:'+str(directory_n)+' picture:'+str(picture_n)
    data_label = label_handling(directory_n,picture_n)

    path = p.data_path()+data_label[0]+'/dp'+data_label[0]+data_label[1]+'r.png'

    model = nn.CNN_classification()
    serializers.load_npz('cnn.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    search_area,rec_area = po.find_object_from_RGB(data_label)

    while(1):

        rec,center,angle,w,p1,p2,p3,p4 = rr.random_rec(search_area,rec_area,scale)
        x,img = generate_input_data(path,rec,scale)
        test_output = model.forward(chainer.Variable(x))
        test_label = np.argmax(test_output.data[0])

        if test_label == 1:
            break

    angle = round(angle*180/np.pi,2)
    z = calculate_z(path,rec,scale)
    print '\n'+'(xc,yc): '+str(center)+',  zc[mm]: '+str(z)
    print 'theta[deg]: '+str(angle)+',  gripper_width: '+str(w)+'\n'
    draw_object_area(img,search_area,rec_area)
    draw_grasp_rectangle(img,rec,scale)
    plt.show()
