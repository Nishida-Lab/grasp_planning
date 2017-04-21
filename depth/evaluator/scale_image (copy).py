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
def load_picture(data_label,scale):

    img =Image.open(p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png')

    resize_img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(resize_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[2]*img_shape[1]*img_shape[0],1))
    img_list = []
    for i in range(len(img_array)):
        img_list.append(img_array[i][0]/255.0)
        #img_list.append(img_array[i][0])

    return img_list


#main
if __name__ == '__main__':

    #dlabel_1 = 1
    #dlabel_2 = 0

    dlabel_1 = np.random.randint(8) + 1
    dlabel_2 = np.random.randint(99) + 1


    data_label = label_handling(dlabel_1,dlabel_2)

    img = load_picture(data_label,4)
    img = np.array(img)*255
    img = np.reshape(img,(120,160,3))

    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()

    print img
