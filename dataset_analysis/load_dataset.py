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
#data_label_1: directory label 1-9
#data_label_2: picture label 1-99
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
    for line in open('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'cpos.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    #load image and draw rectangles
    image = Image.open('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'r.png')
    draw = ImageDraw.Draw(image)

    for i in range(len(xy_data)):
        draw.line((xy_data[i][0],xy_data[i][1])+(xy_data[i][2],xy_data[i][3]), fill='yellow', width=2)
        draw.line((xy_data[i][2],xy_data[i][3])+(xy_data[i][4],xy_data[i][5]), fill='green', width=2)
        draw.line((xy_data[i][4],xy_data[i][5])+(xy_data[i][6],xy_data[i][7]), fill='yellow', width=2)
        draw.line((xy_data[i][6],xy_data[i][7])+(xy_data[i][0],xy_data[i][1]), fill='green', width=2)

    #set image label
    image_label ='directory:' + str(data_label_1) + ' picture:' + str(data_label_2)
    #draw = ImageDraw.Draw(image)
    draw.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
    #draw.text((300,430), image_label, (255, 0, 0))

    #show point cloud graph
    load_point_cloud(data_label_1,data_label_2,image_label)

    #show image
    image.show()

    return xy_data


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
    for line in open('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'cneg.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    #load image and draw rectangles
    image = Image.open('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'r.png')
    draw = ImageDraw.Draw(image)

    for i in range(len(xy_data)):
        draw.line((xy_data[i][0],xy_data[i][1])+(xy_data[i][2],xy_data[i][3]), fill='red', width=2)
        draw.line((xy_data[i][2],xy_data[i][3])+(xy_data[i][4],xy_data[i][5]), fill='blue', width=2)
        draw.line((xy_data[i][4],xy_data[i][5])+(xy_data[i][6],xy_data[i][7]), fill='red', width=2)
        draw.line((xy_data[i][6],xy_data[i][7])+(xy_data[i][0],xy_data[i][1]), fill='blue', width=2)

    #set image label
    image_label ='directory:' + str(data_label_1) + ' picture:' + str(data_label_2)
    #draw = ImageDraw.Draw(image)
    draw.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
    #draw.text((300,430), image_label, (255, 0, 0))

    #show point cloud graph
    load_point_cloud(data_label_1,data_label_2,image_label)

    #show image
    image.show()

    return xy_data


#load point cloud data
def load_point_cloud(data_label_1,data_label_2,image_label):

    shutil.copy('../../grasp_dataset/'+data_label_1+'/pcd'+data_label_1+data_label_2+'.txt','pcd_data'+data_label_1+data_label_2+'.pcd')
    point_cloud_raw = pcl.load('pcd_data'+data_label_1+data_label_2+'.pcd')
    os.remove('pcd_data'+data_label_1+data_label_2+'.pcd')
    point_cloud = np.asarray(point_cloud_raw)

    x = []
    y = []
    z = []

    for i in range(0,len(point_cloud),100):
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    i = data_label_1 + '-' + data_label_2

    fig = plt.figure(i)
    ax = Axes3D(fig)
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    #ax.set_title(image_label)
    ax.tick_params(labelsize = 15)
    ax.set_xlabel(r'$x$'+' [mm]',fontsize=18)
    ax.set_xticks([i for i in range(600,2100,300)])
    ax.set_ylabel(r'$y$'+' [mm]',fontsize=18)
    ax.set_yticks([i for i in range(-600,1400,300)])
    ax.set_zlabel(r'$z$'+' [mm]',fontsize=18)
    ax.set_zticks([i for i in range(-150,150,50)])


#main
if __name__ == '__main__':

    #demo
    dlabel_1 = 7
    dlabel_2 = 32

    #dlabel_1 = np.random.randint(8) + 1
    #dlabel_2 = np.random.randint(99) + 1

    print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)
    positive_vartices = load_positive_dataset(dlabel_1,dlabel_2)
    negative_vartices = load_negative_dataset(dlabel_1,dlabel_2)

    # for i in range(1):
    #     dlabel_1 = np.random.randint(8) + 1
    #     dlabel_2 = np.random.randint(99) + 1
    #     print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)
    #     positive_vartices = load_positive_dataset(dlabel_1,dlabel_2)
    #     negative_vartices = load_negative_dataset(dlabel_1,dlabel_2)

    plt.show()
