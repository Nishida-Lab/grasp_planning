# -*- coding: utf-8 -*-

#python library
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
import sys
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
import random

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers

#pygame library
import pygame
from pygame.locals import *

#python script
import network_structure as nn
import pickup_object as po
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
def load_picture(path,scale):

    img =Image.open(path)

    resize_img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(resize_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[2]*img_shape[1]*img_shape[0],1))
    img_list = []
    for i in range(len(img_array)):
        img_list.append(img_array[i][0]/255.0)

    return img_list


# generate grasp rectangle randomly
def random_rec(object_area,scale):

    theta = random.uniform(-1,1) * np.pi

    if len(object_area)==0:
        area_index = 0
    else:
        area_index = randint(0,len(object_area))

    xc_yc = []

    xc = randint(object_area[area_index][0],object_area[area_index][0]+object_area[area_index][2])
    yc = randint(object_area[area_index][1],object_area[area_index][1]+object_area[area_index][3])

    xc_yc.append(xc)
    xc_yc.append(yc)

    a = randint(30,80)
    #b = randint(10,30)
    b = 20

    x1 = a*np.cos(theta)-b*np.sin(theta)+xc
    y1 = a*np.sin(theta)+b*np.cos(theta)+yc

    x2 = -a*np.cos(theta)-b*np.sin(theta)+xc
    y2 = -a*np.sin(theta)+b*np.cos(theta)+yc

    x3 = -a*np.cos(theta)+b*np.sin(theta)+xc
    y3 = -a*np.sin(theta)-b*np.cos(theta)+yc

    x4 = a*np.cos(theta)+b*np.sin(theta)+xc
    y4 = a*np.sin(theta)-b*np.cos(theta)+yc

    rec_list = []

    rec_list.append(round(x1/scale,2))
    rec_list.append(round(y1/scale,2))

    rec_list.append(round(x2/scale,2))
    rec_list.append(round(y2/scale,2))

    rec_list.append(round(x3/scale,2))
    rec_list.append(round(y3/scale,2))

    rec_list.append(round(x4/scale,2))
    rec_list.append(round(y4/scale,2))

    return rec_list,xc_yc,theta


# generate input data for CNN
def input_data(path,rec_list,scale):
    img_list = load_picture(path,scale)
    x = rec_list + img_list
    x = np.array(x,dtype=np.float32).reshape((1,57608))
    return x


# draw grasp rectangle
def draw_grasp_rectangle(color1,color2):
    pygame.draw.line(screen, rec_color1, (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
    pygame.draw.line(screen, rec_color2, (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
    pygame.draw.line(screen, rec_color1, (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
    pygame.draw.line(screen, rec_color2, (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)


# write text
def captions(dir_n,pic_n,mss,rc,cnt,rad,f1,f2,f3):
    text1 = f1.render("directory_n: "+str(dir_n)+" picture_n: "+str(pic_n), True, (255,255,255))
    text2 = f1.render("quit: ESC", True, (255,255,255))
    text3 = f2.render(mss, True, (255,0,0))
    text4 = f3.render("rectangle(scaled):", True, (255,0,0))
    text5 = f3.render("  "+str(rc), True, (255,0,0))
    text6 = f3.render("center_point: "+str(cnt)+",  angle [deg]: "+str(round(rad*(180/np.pi),2)), True, (255,0,0))
    screen.blit(text1, [20, 20])
    screen.blit(text2, [20, 50])
    screen.blit(text3, [80, 80])
    screen.blit(text4, [20, 400])
    screen.blit(text5, [20, 420])
    screen.blit(text6, [20, 450])


#main
if __name__ == '__main__':

    #directory_n = 3
    #picture_n = 77

    #demo
    directory_n = 5
    picture_n = 77

    # random checking
    #directory_n = randint(7)+1
    #picture_n = randint(98)+1

    # multiple object recrangles will be appeard
    #directory_n = 7
    #picture_n = 80

    # "yellow plate"
    #directory_n = 3
    #picture_n = 88

    scale = 4

    print 'directory:'+str(directory_n)+' picture:'+str(picture_n)
    data_label = label_handling(directory_n,picture_n)
    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    screen_size = (640, 480)
    pygame.init()
    pygame.display.set_mode(screen_size)
    pygame.display.set_caption("random search")

    screen = pygame.display.get_surface()
    bg = pygame.image.load(path).convert_alpha()
    rect_bg = bg.get_rect()

    pygame.font.init()
    font1 = pygame.font.Font(None, 30)
    font2 = pygame.font.Font(None, 40)
    font3 = pygame.font.Font(None, 25)

    search_area = po.find_object(path)

    while (1):

        rec,center,angle = random_rec(search_area,scale)
        x = input_data(path,rec,scale)

        test_output = model.forward(chainer.Variable(x))
        test_label = np.argmax(test_output.data[0])

        pygame.display.update()
        pygame.time.wait(500)
        screen.fill((0, 0, 0))
        screen.blit(bg, rect_bg)

        # draw object area
        for i in range(len(search_area)):
            pygame.draw.rect(screen, (120,120,255), Rect(search_area[i]),3)

        # draw grasp rectangle
        if test_label == 1:
            rec_color1 = (255,255,0)
            rec_color2 = (0,255,0)
            message = "evaluation: graspable"
        else:
            rec_color1 = (255,0,0)
            rec_color2 = (0,0,255)
            message = "evaluation: non-graspable"

        draw_grasp_rectangle(rec_color1,rec_color2)
        captions(directory_n,picture_n,message,rec,center,angle,font1,font2,font3)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
