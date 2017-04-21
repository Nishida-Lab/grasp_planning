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

    data_n = 10

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

    print "total: " + str(len(X))

    indexes = np.random.permutation(len(X))

    X_data = []
    Y_data = []

    for i in range(data_n):
        X_data.append(X[indexes[i]])
        Y_data.append(Y[indexes[i]])

    X_data = np.array(X_data, dtype = np.float32)
    Y_data = np.array(Y_data)

    return X_data,Y_data


# view dataset
def dataset_viewer(dataset,scale):

    n = len(dataset)

    img = []
    rec = []
    y = []

    for i in range(n):

        x = dataset[i][0]

        l = len(x)

        rec.append(x[0:8])
        img.append(x[8:l])
        y.append(dataset[i][1])

        img_array = np.array(img[i]).reshape(330/scale,400/scale)
        img_array = img_array *255

        image = Image.fromarray(np.uint8(img_array))
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)

        color = ['red', 'blue']

        if y[i] == 0:
            color = ['red', 'blue']
        else:
            color = ['yellow', 'green']

        draw.line((rec[i][0],rec[i][1])+(rec[i][2],rec[i][3]), fill=color[0], width=2)
        draw.line((rec[i][2],rec[i][3])+(rec[i][4],rec[i][5]), fill=color[1], width=2)
        draw.line((rec[i][4],rec[i][5])+(rec[i][6],rec[i][7]), fill=color[0], width=2)
        draw.line((rec[i][6],rec[i][7])+(rec[i][0],rec[i][1]), fill=color[1], width=2)
        image.show()
        image = 0

    return img,rec


#main
if __name__ == '__main__':

    dir_n = 2
    img_n = 99

    scale = 1

    label = label_handling(dir_n,img_n)

    img = load_depth_image(label,scale)
    rec = load_rectangle(label,0,scale)

    x,y = prepare_dataset(dir_n,dir_n,img_n,scale)

    sample_data = zip(x,y)

    dataset_viewer(sample_data,scale)

    plt.show()
