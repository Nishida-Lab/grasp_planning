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

#python script
import path as p


#label preparation
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


#load rectangles
def load_rec(path,neg_pos):
    rec_data = []
    for line in open(path+'c'+neg_pos+'.txt').readlines():
        rec_str = line.split(' ')
        rec_data.append([float(rec_str[0]),float(rec_str[1])])
    rec_data = np.array(rec_data).reshape(len(rec_data)/4,8)
    return rec_data


#draw rectangles
def draw_rectangle(draw,xy_data,color):
    for i in range(len(xy_data)):
        draw.line((xy_data[i][0],xy_data[i][1])+(xy_data[i][2],xy_data[i][3]), fill=color[0], width=2)
        draw.line((xy_data[i][2],xy_data[i][3])+(xy_data[i][4],xy_data[i][5]), fill=color[1], width=2)
        draw.line((xy_data[i][4],xy_data[i][5])+(xy_data[i][6],xy_data[i][7]), fill=color[0], width=2)
        draw.line((xy_data[i][6],xy_data[i][7])+(xy_data[i][0],xy_data[i][1]), fill=color[1], width=2)


#load point cloud data
def load_point_cloud(data_label):

    point_cloud = []

    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt'

    num_lines = sum(1 for line in open(path))

    for i in range(11,num_lines):

        line = linecache.getline(path, i)
        xy_str = line.split(' ')

        point_cloud.append([float(xy_str[0]),float(xy_str[1]),float(xy_str[2])])

    point_cloud = np.asarray(point_cloud)

    x = []
    y = []
    z = []

    for i in range(0,len(point_cloud),100):
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    fig = plt.figure(0)
    ax = Axes3D(fig)
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    ax.tick_params(labelsize = 15)
    ax.set_xlabel(r'$x$'+' [mm]',fontname='roman',fontsize=18)
    ax.set_xticks([i for i in range(600,2100,300)])
    ax.set_ylabel(r'$y$'+' [mm]',fontname='roman',fontsize=18)
    ax.set_yticks([i for i in range(-600,1400,300)])
    ax.set_zlabel(r'$z$'+' [mm]',fontname='roman',fontsize=18)
    ax.set_zticks([i for i in range(-150,150,50)])


#load dataset and draw grasp rectangle
#data_label_1: directory label 1-9
#data_label_2: picture label 1-99
def load_dataset(directory_n,picture_n):

    neg_pos = ['neg','pos']

    data_label = label_handling(directory_n,picture_n)
    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]

    #load image and draw rectangles
    # for negative rectangles
    image1 = Image.open(path+'r.png')
    draw1 = ImageDraw.Draw(image1)
    neg_rec = load_rec(path,neg_pos[0])
    neg_color = ['red', 'blue']
    draw_rectangle(draw1,neg_rec,neg_color)
    image1.show()

    # for positive rectangles
    image2 = Image.open(path+'r.png')
    draw2 = ImageDraw.Draw(image2)
    pos_rec = load_rec(path,neg_pos[1])
    pos_color = ['yellow', 'green']
    draw_rectangle(draw2,pos_rec,pos_color)
    image2.show()

    #show point cloud graph
    load_point_cloud(data_label)

    return neg_rec,pos_rec


#main
if __name__ == '__main__':

    # demo
    #dlabel_1 = 7
    #dlabel_2 = 32

    # key input
    dlabel_1 = input('Directory No > ')
    dlabel_2 = input('Image No > ')

    # random
    #dlabel_1 = np.random.randint(8) + 1
    #dlabel_2 = np.random.randint(99) + 1

    print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)
    neg_rec,pos_rec = load_dataset(dlabel_1,dlabel_2)

    plt.show()
