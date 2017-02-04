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

    plt.figure(2)
    plt.imshow(cv2.cvtColor(grayed, cv2.COLOR_GRAY2RGB))


#load point cloud data
def load_point_cloud(data_label):

    shutil.copy('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt','pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud_raw = pcl.load('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    os.remove('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud = np.asarray(point_cloud_raw)

    x = []
    y = []
    z = []
    xyz = []

    for i in range(0,len(point_cloud),100):
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    for i in range(0,len(point_cloud)):
        xyz.append([point_cloud[i][0],point_cloud[i][1],point_cloud[i][2]])

    xyz = np.array(xyz)

    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    ax1.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    #p1 = ax0.scatter3D(x,y,z,color=(1.0,0,0),marker='o',s=1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    return x,y,z


#generate depth iamge
def depth_image(x,y,z):

    xi = []
    yi = []
    zi = []
    dimg = np.zeros((len(x)+1,len(y)+1))

    x_diff = max(x)-min(x)
    y_diff = max(y)-min(y)
    z_diff = max(z)-min(z)

    for i in range(len(x)):
        xi.append(int(((x[i]-min(x))/x_diff)*len(x)))
        yi.append(int(((y[i]-min(y))/y_diff)*len(y)))
        zi.append(int(((z[i]-min(z))/z_diff)*255))

    # --------------------------------------------------
    #for i in range(len(x)-1):
     #   dimg[xi[i]][yi[i]] = zi[i]

    print zi
    print dimg
    print dimg[xi[100]][yi[100]]
    dimg = cv2.resize(dimg,(dimg.shape[0]/20,dimg.shape[0]/20))

    plt.figure(3)
    plt.imshow(dimg)
    plt.gray()
    # cv2.imshow('result', dimg), cv2.waitKey(0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    fig = plt.figure(4)
    ax = Axes3D(fig)
    ax.plot_trisurf(xi, yi, zi, cmap=cm.jet, linewidth=0.2)
    #p1 = ax1.scatter3D(xi,yi,zi,color=(1.0,0,0),marker='o',s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


#main
if __name__ == '__main__':

    #demo
    #dlabel_1 = 5
    #dlabel_2 = 75

    #dlabel_1 = 7
    #dlabel_2 = 32

    dlabel_1 = np.random.randint(8) + 1
    dlabel_2 = np.random.randint(99) + 1

    data_label = label_handling(dlabel_1,dlabel_2)

    print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)

    load_rgb_image(data_label)
    x,y,z = load_point_cloud(data_label)
    depth_image(x,y,z)
    plt.show()
