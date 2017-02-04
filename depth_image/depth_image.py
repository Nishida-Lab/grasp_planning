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


# load RGB image
def load_rgb_image(data_label):

    path = '../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
    img = cv2.imread(path)

    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.figure(0)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')


#load point cloud data
def load_point_cloud(data_label):

    shutil.copy('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt','pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud_raw = pcl.load('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    os.remove('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud = np.asarray(point_cloud_raw)

    x = []
    y = []
    z = []
    xd = []
    yd = []
    zd = []
    xyz = []

    for i in range(0,len(point_cloud),10):
        if 750 < point_cloud[i][0] and point_cloud[i][0] < 1400 and -400 < point_cloud[i][1] and point_cloud[i][1] < 600:
            x.append(point_cloud[i][0])
            y.append(point_cloud[i][1])
            z.append(point_cloud[i][2])

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

    return x,y,z


#generate depth iamge
def depth_image(x,y,z):

    xi = []
    yi = []
    zi = []

    im_size = 300
    img = np.zeros((im_size,im_size,3),np.uint8)
    dimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    x_diff = max(x)-min(x)
    y_diff = max(y)-min(y)
    z_diff = max(z)-min(z)
    im_size -= 1

    for i in range(len(x)):
        xin = int(((x[i]-min(x))/x_diff)*(im_size))
        yin = int(((y[i]-min(y))/y_diff)*(im_size))
        zv = int(((z[i]-min(z))/z_diff)*255)
        dimg[xin][yin] = zv
        print "progress: "+str(i)+"/"+str(len(x))

    center = tuple(np.array([dimg.shape[1] * 0.5, dimg.shape[0] * 0.5]))
    size = tuple(np.array([dimg.shape[1], dimg.shape[0]]))
    angle = 180.0
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    dimg = cv2.warpAffine(dimg, rotation_matrix, size, flags=cv2.INTER_CUBIC)

    return dimg


#main
if __name__ == '__main__':

    #demo
    #dlabel_1 = 5
    #dlabel_2 = 75

    #dlabel_1 = 2
    #dlabel_2 = 68

    dlabel_1 = np.random.randint(8) + 1
    dlabel_2 = np.random.randint(99) + 1

    data_label = label_handling(dlabel_1,dlabel_2)

    # load rgb image
    load_rgb_image(data_label)

    # load point cloud
    x,y,z = load_point_cloud(data_label)

    # generate depth image from point cloud
    img = depth_image(x,y,z)

    # save depth image
    name = '../../grasp_dataset/'+data_label[0]+'/dm'+data_label[0]+data_label[1]+'r.png'
    #cv2.imwrite(name,img)
    #print 'saved depth image'

    print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)

    plt.figure(2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.axis('off')

    plt.show()
