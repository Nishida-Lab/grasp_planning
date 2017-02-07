#python library
import numpy as np
from PIL import Image
import ImageDraw
import os


#just show the image
def show_the_image():
    image = Image.open('../../grasp_dataset/01/pcd0101r.png')
    image.show()


#draw grasp rectangle(prototype)
def draw_rectangle_prototype():
    image = Image.open('../../grasp_dataset/01/pcd0101r.png')
    draw = ImageDraw.Draw(image)
    draw.line((253,319.7)+(309,324), fill='yellow', width=2)
    draw.line((309,324)+(307,350), fill='green', width=2)
    draw.line((307,350)+(251,345.7), fill='yellow', width=2)
    draw.line((251,345.7)+(253,319.7), fill='green', width=2)
    image.show()


#read dataset and draw grasp rectangle
def read_outputs():

    xy_data = []

    for line in open('../../grasp_dataset/01/pcd0101cpos.txt').readlines():
        xy_str = line.split(' ')
        xy_data.append([float(xy_str[0]),float(xy_str[1])])

    xy_data = np.array(xy_data).reshape(len(xy_data)/4,8)

    return xy_data


#draw grasp rectangle
def draw_rectangle(xy_data):

    image = Image.open('../../grasp_dataset/01/pcd0101r.png')
    draw = ImageDraw.Draw(image)

    for i in range(len(xy_data)):
        draw.line((xy_data[i][0],xy_data[i][1])+(xy_data[i][2],xy_data[i][3]), fill='yellow', width=2)
        draw.line((xy_data[i][2],xy_data[i][3])+(xy_data[i][4],xy_data[i][5]), fill='green', width=2)
        draw.line((xy_data[i][4],xy_data[i][5])+(xy_data[i][6],xy_data[i][7]), fill='yellow', width=2)
        draw.line((xy_data[i][6],xy_data[i][7])+(xy_data[i][0],xy_data[i][1]), fill='green', width=2)

    image.show()


#main
if __name__ == '__main__':
    #show_the_image()
    #draw_rectangle_prototype()
    vartices = read_outputs()
    print vartices.shape
    draw_rectangle(vartices)
