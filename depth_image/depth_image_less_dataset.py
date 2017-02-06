#python library
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# OpenCV
import cv2

#PCL
import pcl


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

    shutil.copy('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt','pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud_raw = pcl.load('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    os.remove('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud = np.asarray(point_cloud_raw)

    x = []
    y = []
    z = []

    for i in range(0,len(point_cloud),2):
        if 750 < point_cloud[i][0] and point_cloud[i][0] < 1275 and -300 < point_cloud[i][1] and point_cloud[i][1] < 400 and 0 < point_cloud[i][2]:
            x.append(point_cloud[i][0])
            y.append(point_cloud[i][1])
            z.append(point_cloud[i][2])

    return x,y,z


#generate depth iamge
def depth_image(x,y,z,data_label):

    xi = []
    yi = []
    zi = []
    x_size = 120
    y_size = 160
    img = np.zeros((x_size,y_size,3),np.uint8)
    dimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    x_diff = max(x)-min(x)
    y_diff = max(y)-min(y)
    z_diff = 100 # maximum object hight: 100[mm]
    x_size -= 1
    y_size -= 1

    for i in range(len(x)):
        xin = ((x[i]-min(x))/x_diff)*x_size
        yin = ((y[i]-min(y))/y_diff)*y_size
        zv = ((z[i]-min(z))/z_diff)*255
        dimg[xin][yin] = zv
        print str(dlabel_1)+'-'+str(dlabel_2)+","+str(i)+"/"+str(len(x))

    # rotate 180
    center = tuple(np.array([dimg.shape[1] * 0.5, dimg.shape[0] * 0.5]))
    size = tuple(np.array([dimg.shape[1], dimg.shape[0]]))
    angle = 180
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    dimg = cv2.warpAffine(dimg, rotation_matrix, size, flags=cv2.INTER_CUBIC)

    return dimg


#main
if __name__ == '__main__':

    l1_min = input('directory_range_min > ')
    l1_max = input('directory_range_max > ')
    l2_min = input('image_range_min > ')
    l2_max = input('image_range_max > ')

    for dlabel_1 in range(l1_min,l1_max+1):
        for dlabel_2 in range(l2_min,l2_max+1):

            data_label = label_handling(dlabel_1,dlabel_2)

            # load point cloud
            x,y,z = load_point_cloud(data_label)

            # generate depth image from point cloud
            img = depth_image(x,y,z,data_label)

            # save depth image
            name = '../../grasp_dataset/'+data_label[0]+'/dp'+data_label[0]+data_label[1]+'r.png'
            cv2.imwrite(name,img)
            print 'saved depth image'

    print 'finished!'
