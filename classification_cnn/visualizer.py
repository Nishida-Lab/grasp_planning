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

# PCL
import pcl


# draw rectangle
def draw_rec(x,y):

    x_shape = x[0].shape

    img_data = np.reshape(x[0],(x_shape[2],x_shape[1],x_shape[0]))
    img_data = np.uint8(img_data)
    img = Image.fromarray(img_data)
    img.show()

    draw_neg = ImageDraw.Draw(img)

    if y == 0:
        draw_neg.line((x[1][0],x[1][1])+(x[1][2],x[1][3]), fill='red', width=2)
        draw_neg.line((x[1][2],x[1][3])+(x[1][4],x[1][5]), fill='blue', width=2)
        draw_neg.line((x[1][4],x[1][5])+(x[1][6],x[1][7]), fill='red', width=2)
        draw_neg.line((x[1][6],x[1][7])+(x[1][0],x[1][1]), fill='blue', width=2)
    elif y == 1:
        draw_neg.line((x[1][0],x[1][1])+(x[1][2],x[1][3]), fill='yellow', width=2)
        draw_neg.line((x[1][2],x[1][3])+(x[1][4],x[1][5]), fill='green', width=2)
        draw_neg.line((x[1][4],x[1][5])+(x[1][6],x[1][7]), fill='yellow', width=2)
        draw_neg.line((x[1][6],x[1][7])+(x[1][0],x[1][1]), fill='green', width=2)
    img.show()



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
