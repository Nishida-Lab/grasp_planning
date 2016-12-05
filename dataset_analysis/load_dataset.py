#python library
import numpy as np
from PIL import Image
import ImageDraw


#load dataset and draw grasp rectangle
#data_label_1: directly label 1-9
#data_label_2: picture label 1-99
def load_dataset(data_label_1,data_label_2):

    #Label preparation for directly operation
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

    image.show()

    return xy_data


#main
if __name__ == '__main__':

    for i in range(3):
        dlabel_1 = np.random.randint(9) + 1
        dlabel_2 = np.random.randint(99) + 1
        print 'data_label_1: '+str(dlabel_1)+' data_label_2: '+str(dlabel_2)
        vartices = load_dataset(dlabel_1,dlabel_2)
