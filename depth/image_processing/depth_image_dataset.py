#python library
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import linecache

# python scripts
import path as p

# OpenCV
import cv2


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


#load point cloud data
def load_point_cloud(data_label):

    point_cloud = []

    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt'

    num_lines = sum(1 for line in open(path))

    for i in range(11,num_lines):
        line = linecache.getline(path, i)
        xy_str = line.split(' ')

        idx = int(xy_str[4])
        row = int(np.floor(idx/640))+1
        col = np.mod(idx,640)+1

        point_cloud.append([float(xy_str[0]),float(xy_str[1]),float(xy_str[2]),row,col])

    point_cloud = np.asarray(point_cloud)

    x = []
    y = []
    z = []
    index = []

    xd = []
    yd = []
    zd = []
    xyz = []

    # pack data
    res = 1
    for i in range(0,len(point_cloud),res):
        if 100 < point_cloud[i][3] and point_cloud[i][3] < 430 and 100 < point_cloud[i][4] and point_cloud[i][4] < 500 and 0 < point_cloud[i][2]:
            z.append(point_cloud[i][2])
            index.append([point_cloud[i][3],point_cloud[i][4]])

    # draw 3-D graph
    for i in range(0,len(point_cloud),100):
        xd.append(point_cloud[i][0])
        yd.append(point_cloud[i][1])
        zd.append(point_cloud[i][2])

    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    ax1.plot_trisurf(xd, yd, zd, cmap=cm.jet, linewidth=0.2)
    #p1 = ax1.scatter3D(xd,yd,zd,color=(1.0,0,0),marker='o',s=1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    return z,index


#generate depth iamge
def depth_image(z,index,data_label):

    x_size = 480
    y_size = 640
    img = np.zeros((x_size,y_size,3),np.uint8)
    dimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    z_diff = 150 # maximum object hight: 150[mm]

    for i in range(len(z)):
        zv = ((z[i]-min(z))/z_diff)*255
        dimg[index[i][0]-1][index[i][1]-1] = zv

        if i%10000 == 0:
            print str(data_label[0])+'-'+str(data_label[1])+","+str(i)+"/"+str(len(z))

    return dimg[100:430,100:500]


#main
if __name__ == '__main__':

    l1_min = input('directory_range_min > ')
    l1_max = input('directory_range_max > ')
    l2_min = input('image_range_min > ')
    l2_max = input('image_range_max > ')

    for dlabel_1 in range(l1_min,l1_max+1):
        for dlabel_2 in range(l2_min,l2_max+1):

            data_label = label_handling(dlabel_1,dlabel_2)

            print 'data_label_1: '+str(data_label[0])+' data_label_2: '+str(data_label[0])

            # load point cloud
            z,index = load_point_cloud(data_label)

            # generate depth image from point cloud
            img = depth_image(z,index,data_label)

            # save depth image
            name = p.data_path()+data_label[0]+'/dp'+data_label[0]+data_label[1]+'r.png'
            cv2.imwrite(name,img)
            print 'saved depth image'

    print 'finished!'
