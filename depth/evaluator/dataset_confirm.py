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

# Python scripts
import visualizer as v
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


# load picture data
def load_depth_image(data_label,scale):

    path = p.data_path()+data_label[0]+'/dp'+data_label[0]+data_label[1]+'r.png'

    img =Image.open(path)

    img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(img,dtype=np.float32)

    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[1]*img_shape[0],1))
    img_list = []

    for i in range(len(img_array)):
       img_list.append(img_array[i][0]/255.0)

    return img_list


# load rectangle data
def load_rectangle(data_label,neg_or_pos,scale):

    neg_pos = ['neg','pos']
    rec_list = []

    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+neg_pos[neg_or_pos]+'.txt'

    for line in open(path).readlines():
        rec_str = line.split(' ')
        x = float(rec_str[0])-100.0
        y = float(rec_str[1])-100.0
        rec_list.append([x/scale,y/scale])

    rec_array = np.array(rec_list,dtype=np.float32).reshape(len(rec_list)/4,8)

    return rec_array


# view dataset
def dataset_viewer(dataset):


#main
if __name__ == '__main__':

    scale = 1
    dir_n = 10
    img_n = 10

    label = label_handling(dir_n,img_n)

    img = load_depth_image(label,scale)
    rec = load_rectangle(label,0,scale)

    print len(img)
