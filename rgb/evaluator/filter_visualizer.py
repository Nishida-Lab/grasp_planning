#python library
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
from PIL import Image, ImageDraw, ImageFont

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers

#python script
import network_structure as nn
import visualizer as v
import path as p

# OpenCV
import cv2



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
def load_picture(data_label,scale):

    img =Image.open(p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png')

    resize_img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(resize_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[2]*img_shape[1]*img_shape[0],1))
    img_list = []
    for i in range(len(img_array)):
        img_list.append(img_array[i][0]/255.0)

    return img_list


# load rectangle data
def load_rectangle(data_label,neg_or_pos,scale):

    neg_pos = ['neg','pos']
    rec_list = []

    for line in open(p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+neg_pos[neg_or_pos]+'.txt').readlines():
        rec_str = line.split(' ')
        x = float(rec_str[0])
        y = float(rec_str[1])
        rec_list.append([x/scale,y/scale])
        #rec_list.append([x,y])

    rec_array = np.array(rec_list,dtype=np.float32).reshape(len(rec_list)/4,8)
    #print rec_array

    return rec_array


# prepare dataset
def prepare_dataset(min_directly_n,max_directly_n,max_picture_n,scale):

    X = []
    Y = []

    for i in range(min_directly_n,max_directly_n+1):
        print "directly No: " + str(i)
        for j in range(max_picture_n+1):
            for neg_pos in range(2):

                data_label = label_handling(i,j)

                img_list = load_picture(data_label,scale)
                rec_array = load_rectangle(data_label,neg_pos,scale)

                for k in range(len(rec_array)):
                    rec_list = rec_array[k].tolist()
                    X.append(rec_list + img_list)
                    Y.append(neg_pos)

    Y = np.array(Y,dtype = np.int32)

    return X,Y


# draw filter
def draw_fil(data,size):
    #plt.style.use('fivethirtyeight')
    tate = len(data)
    yoko = len(data)/21 + 1
    #fig = plt.figure(figsize=(tate+2, yoko))
    fig = plt.figure(figsize=(tate, yoko))
    for t, x in enumerate(data):
        plt.gray()
        ax = fig.add_subplot(yoko , 3, t+1 )
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.tick_params(labelbottom="off")
        ax.tick_params(labelleft="off")
        xx = np.array(range(size + 1))
        yy = np.array(range(size, -1, -1))
        X, Y = np.meshgrid(xx, yy)
        ax.pcolor(X,Y,x[0])


#main
if __name__ == '__main__':

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    dlabel_1 = input('Directory No > ')
    dlabel_2 = input('Image No > ')
    data_label = label_handling(dlabel_1,dlabel_2)

    # load picture
    img = load_picture(data_label,4)
    img = np.array(img,dtype=np.float32).reshape(1,3,120,160)

    # view original image (scaled)
    image = img.reshape(120,160,3)*225
    image = Image.fromarray(np.uint8(image))
    image.show()
    image.save('filter/original.png')

    # view conv filter
    i_cw1 = model.conv.W.data
    #print i_cw1.shape
    # for i in range(len(i_cw1)):
    #     flt = i_cw1[i]
    #     flt = np.array(flt).reshape(5,5,3)
    #     print flt
    #     flt = Image.fromarray(np.uint8(flt))
    #     flt.show()

    # view conv1 filtered image
    cv1 = model.conv(img).data
    print cv1.shape
    im_cv1 = cv1.reshape(116,156,3)*255
    im_cv1 = Image.fromarray(np.uint8(im_cv1)).transpose(3)
    im_cv1.show()
    im_cv1.save('filter/conv1.png')

    # view after max pooling
    p1 = F.max_pooling_2d(model.conv(img),2,stride = 2).data
    print p1.shape
    im_p1 = p1.reshape(58,78,3) * 225
    im_p1 = Image.fromarray(np.uint8(im_p1))
    im_p1.show()
    im_p1.save('filter/pool1.png')

    plt.show()
