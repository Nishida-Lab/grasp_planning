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

# Python scripts
import visualizer as v


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


# load rectangle data
def load_rectangle(data_label,neg_or_pos,scale):

    neg_pos = ['neg','pos']
    xy_rec = []

    for line in open('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+neg_pos[neg_or_pos]+'.txt').readlines():
        xy_str = line.split(' ')
        x = float(xy_str[0])
        y = float(xy_str[1])
        xy_rec.append([round((x/scale),2),round((y/scale),2)])

    xy_rec = np.array(xy_rec,dtype=np.float32).reshape(len(xy_rec)/4,8)

    return xy_rec


# merge picture and rectangle
def load_picture(data_label,scale):

    img =Image.open('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png')

    resize_img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(resize_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[2],img_shape[1],img_shape[0]))

    return img_array


# prepare dataset
def prepare_dataset(min_directly_n,max_directly_n,max_picture_n,scale):

    X = []
    Y = []

    for i in range(min_directly_n,max_directly_n+1):
        for j in range(max_picture_n+1):
            for neg_pos in range(2):

                data_label = label_handling(i,j)

                img_array = load_picture(data_label,scale)
                xy_rec = load_rectangle(data_label,neg_pos,scale)

                for k in range(len(xy_rec)):
                    X.append([img_array,xy_rec[k]])
                    Y.append(neg_pos)

    X = np.array(X)
    Y = np.array(Y,dtype = np.int32)

    return X,Y


# generate_dataset
def generate_dataset(train_N,validation_N):

    scale = 4

    min_dir_n = 1
    max_dir_n = 1
    max_pic_n = 99

    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []

    X,Y = prepare_dataset(min_dir_n,max_dir_n,max_pic_n,scale)

    print " "
    print "loaded learning dataset"
    print "directly: " + str(min_dir_n) +"-"+ str(max_dir_n) + " picture: 0-" + str(max_pic_n)
    print "total: " + str(len(X))

    indexes = np.random.permutation(len(X))

    for i in range(train_N + validation_N):

        if i < train_N:
            X_train.append(X[indexes[i]])
            Y_train.append(Y[indexes[i]])
        else:
            X_validation.append(X[indexes[i]])
            Y_validation.append(Y[indexes[i]])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    print "train_N: " + str(len(X_train))
    print "validation_N: " + str(len(X_validation))
    print " "

    v.draw_rec(X_train[0],Y_train[0])

    return X_train,Y_train,X_validation,Y_validation


def test_dataset(test_N):

    scale = 4

    min_dir_n = 9
    max_dir_n = 10
    max_pic_n = 30

    X_test = []
    Y_test = []

    X,Y = prepare_dataset(min_dir_n,max_dir_n,max_pic_n,scale)

    print " "
    print "loaded test dataset"
    print "directly: " + str(min_dir_n) +"-"+ str(max_dir_n) + " picture: 0-" + str(max_pic_n)
    print "total: " + str(len(X))

    indexes = np.random.permutation(len(X))

    for i in range(test_N):
        if i < test_N:
            X_test.append(X[indexes[i]])
            Y_test.append(Y[indexes[i]])
        else:
            break

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print "test_N: " + str(len(X_test))
    print " "

    return X_test,Y_test


#main
if __name__ == '__main__':

    train_N = 500
    validation_N = 50
    test_N = 10

    Xt,Yt,Xv,Yv = generate_dataset(train_N,validation_N)
    print Xt[0].shape
    print Xt[0][0].shape
    #X_test,Y_test = test_dataset(test_N)
