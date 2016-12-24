# python library
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import json


# PCL
import pcl


# separate input data into image and rec
def data_separator(X):

    img = []
    rec = []

    for i in range(len(X)):
        if i < 8:
            rec.append(X[i])
        else:
            img.append(X[i]*255)

    rec = np.asarray(rec).reshape(8,1).astype(np.float32)
    img = np.asarray(img).reshape(3,160,120).astype(np.float32)

    return img,rec


#visualize loss reduction
def loss_visualizer():

    epoch = []
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    f = open('./result/log', 'r') #load log file
    data = json.load(f)
    f.close()

    value = []

    for i in range(0,len(data)):
        value = data[i]
        epoch.append(value["epoch"])
        train_loss.append(value["main/loss"])
        test_loss.append(value["validation/main/loss"])
        train_accuracy.append(value["main/accuracy"])
        test_accuracy.append(value["validation/main/accuracy"])

    plt.figure(1)
    plt.plot(epoch,train_loss,"b",label = "train LOSS")
    plt.plot(epoch,test_loss,"g",label = "validation LOSS")
    plt.yscale('log')
    plt.title("LOSS reduction")
    plt.legend()
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("LOSS", fontsize=20)
    plt.tick_params(labelsize=18)

    plt.figure(2)
    plt.plot(epoch,train_accuracy,"b",label = "train accuracy")
    plt.plot(epoch,test_accuracy,"g",label = "validation accuracy ")
    plt.title("accuracy increase")
    plt.legend(loc = "lower right")
    plt.xlabel("epoch",fontsize=20)
    plt.ylabel("accuracy",fontsize=20)
    plt.tick_params(labelsize=18)

# draw rectangle
def draw_rec(x,estimated,actual):

    zoom = 3
    img,rec = data_separator(x)
    img_shape = img.shape

    img_data = np.reshape(img,(120,160,3))
    img_data = np.uint8(img_data)
    img = Image.fromarray(img_data)
    resize_img = img.resize((img.size[0]*zoom,img.size[1]*zoom))
    #resize_img.show()

    rec = rec*zoom
    draw_rec = ImageDraw.Draw(resize_img)

    if actual == 0:
        draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='red', width=2)
        draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='blue', width=2)
        draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
        draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
        actual_label = 'negative'
    elif actual == 1:
        draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
        draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
        draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='yellow', width=2)
        draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='green', width=2)
        actual_label = 'positive'

    #set image label
    if estimated == 0:
        estimated_label = 'negative'
    else:
        estimated_label = 'positive'

    image_label = " estimated: " + estimated_label + " actual: " + actual_label
    #draw = ImageDraw.Draw(image)
    draw_rec.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
    draw_rec.text((10,10), image_label, (255, 0, 0))

    resize_img.show()



#load point cloud data
def load_point_cloud(data_label):

    image_label ='directly:' + str(data_label[0]) + ' picture:' + str(data_label[1])

    shutil.copy('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt','pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud_raw = pcl.load('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    os.remove('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud = np.asarray(point_cloud_raw)

    x = []
    y = []
    z = []

    for i in range(0,len(point_cloud),100):
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    i = data_label[0] + '-' + data_label[1]

    fig = plt.figure(i)
    ax = Axes3D(fig)
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    ax.set_title(image_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
