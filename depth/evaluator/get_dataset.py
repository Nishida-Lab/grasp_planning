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


# prepare dataset
def prepare_dataset(min_directory_n,max_directory_n,max_picture_n,scale):

    X = []
    Y = []

    for i in range(min_directory_n,max_directory_n+1):

        print "loading from directory No: " + str(i)

        for j in range(max_picture_n+1):
            for neg_pos in range(2):

                data_label = label_handling(i,j)

                img_list = load_depth_image(data_label,scale)
                rec_array = load_rectangle(data_label,neg_pos,scale)

                for k in range(len(rec_array)):
                    rec_list = rec_array[k].tolist()
                    X.append(rec_list + img_list)
                    Y.append(neg_pos)

    Y = np.array(Y,dtype = np.int32)

    return X,Y


# generate train and validation dataset
def generate_dataset(train_N,validation_N,scale):

    min_dir_n = 1
    max_dir_n = 6
    max_pic_n = 99

    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []

    X,Y = prepare_dataset(min_dir_n,max_dir_n,max_pic_n,scale)

    print " "
    print "loaded learning dataset"
    print "directory: " + str(min_dir_n) +"-"+ str(max_dir_n) + " picture: 0-" + str(max_pic_n)
    print "total data amount: " + str(len(X))

    indexes = np.random.permutation(len(X))

    for i in range(train_N + validation_N):

        if i < train_N:
            X_train.append(X[indexes[i]])
            Y_train.append(Y[indexes[i]])
        else:
            X_validation.append(X[indexes[i]])
            Y_validation.append(Y[indexes[i]])

    X_train = np.array(X_train, dtype = np.float32)
    Y_train = np.asarray(Y_train)

    X_validation = np.array(X_validation, dtype = np.float32)
    Y_validation = np.asarray(Y_validation)

    print "train_N: " + str(len(X_train))
    print "validation_N: " + str(len(X_validation))
    print " "

    return X_train,Y_train,X_validation,Y_validation


# generate validation dataset (for conformation after training CNN)
def validation_dataset(validation_N,scale):

    min_dir_n = 1
    max_dir_n = 6
    max_pic_n = 99

    X_v = []
    Y_v = []

    X,Y = prepare_dataset(min_dir_n,max_dir_n,max_pic_n,scale)

    indexes = np.random.permutation(len(X))

    for i in range(validation_N):
        X_v.append(X[indexes[i]])
        Y_v.append(Y[indexes[i]])

    Xv = np.array(X_v, dtype = np.float32)
    Yv = np.array(Y_v)

    print " "
    print "loaded test dataset"
    print "directory: " + str(min_dir_n) +"-"+ str(max_dir_n) + " picture: 0-" + str(max_pic_n)
    print "total: " + str(len(X))

    print "test_N: " + str(len(Xv))
    print " "

    return Xv,Yv



# generate test dataset
# !!!!!!!!!! never use this function !!!!!!!!!!
def test_dataset(test_N,scale):

    min_dir_n = 7
    max_dir_n = 8
    max_pic_n = 30

    X_test = []
    Y_test = []

    X,Y = prepare_dataset(min_dir_n,max_dir_n,max_pic_n,scale)

    indexes = np.random.permutation(len(X))

    for i in range(test_N):
        X_test.append(X[indexes[i]])
        Y_test.append(Y[indexes[i]])

    X_test = np.array(X_test, dtype = np.float32)
    Y_test = np.array(Y_test)

    print " "
    print "loaded test dataset"
    print "directory: " + str(min_dir_n) +"-"+ str(max_dir_n) + " picture: 0-" + str(max_pic_n)
    print "total: " + str(len(X))

    print "test_N: " + str(len(X_test))
    print " "

    return X_test,Y_test


#main
if __name__ == '__main__':

    scale = 2

    train_N = 100
    validation_N = 50
    test_N = 10

    Xtr,Ytr,Xv,Yv = generate_dataset(train_N,validation_N,scale)
    Xte,Yte = test_dataset(test_N,scale)

    train = zip(Xtr,Ytr)
    test = zip(Xte,Yte)
