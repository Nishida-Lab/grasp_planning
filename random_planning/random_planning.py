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

    img =Image.open('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png')

    resize_img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(resize_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[2]*img_shape[1]*img_shape[0],1))
    img_list = []
    for i in range(len(img_array)):
        img_list.append(img_array[i][0]/255.0)

    return img_list


def random_input(data_label):

    scale = 4

    theta = random.uniform(-1,1) * np.pi

    xc = randint(1,680)
    yc = randint(1,480)

    a = randint(1,85)
    b = randint(1,60)

    x1 = -a*np.cos(theta)-b*np.sin(theta)+xc
    y1 = -a*np.sin(theta)+b*np.cos(theta)+yc

    x2 = a*np.cos(theta)-b*np.sin(theta)+xc
    y2 = a*np.sin(theta)+b*np.cos(theta)+yc

    x3 = a*np.cos(theta)+b*np.sin(theta)+xc
    y3 = a*np.sin(theta)-b*np.cos(theta)+yc

    x4 = -a*np.cos(theta)+b*np.sin(theta)+xc
    y4 = -a*np.sin(theta)-b*np.cos(theta)+yc

    rec_list = []

    rec_list.append(x1/scale)
    rec_list.append(y1/scale)

    rec_list.append(x2/scale)
    rec_list.append(y2/scale)

    rec_list.append(x3/scale)
    rec_list.append(y3/scale)

    rec_list.append(x4/scale)
    rec_list.append(y4/scale)

    img_list = load_picture(data_label,4)
    x = rec_list + img_list
    x = np.array(x,dtype=np.float32).reshape((1,57608))
    return x


#main
if __name__ == '__main__':

    directly_n = 1
    picture_n = 2
    scale = 4

    data_label = label_handling(directly_n,picture_n)
    path = '../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    screen_size = (640, 480)
    pygame.init()
    pygame.display.set_mode(screen_size)
    pygame.display.set_caption("random planning")

    screen = pygame.display.get_surface()
    bg = pygame.image.load(path).convert_alpha()
    rect_bg = bg.get_rect()

    while (1):

        x = random_input(data_label)

        test_output = model.forward(chainer.Variable(x))
        test_label = np.argmax(test_output.data[0])

        print test_output.data
        print test_label
        print x[0:8]

        pygame.display.update()
        pygame.time.wait(30)
        screen.fill((0, 0, 0))
        screen.blit(bg, rect_bg)

        if test_label == 1:
            pygame.draw.line(screen, (255,255,0), (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
            pygame.draw.line(screen, (0,255,0), (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
            pygame.draw.line(screen, (255,255,0), (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
            pygame.draw.line(screen, (0,255,0), (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)
            break
        else:
            pygame.draw.line(screen, (255,0,0), (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
            pygame.draw.line(screen, (0,0,255), (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
            pygame.draw.line(screen, (255,0,0), (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
            pygame.draw.line(screen, (0,0,255), (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)

    while(1):

        pygame.display.update()
        pygame.time.wait(30)
        screen.fill((0, 0, 0))
        screen.blit(bg, rect_bg)
        pygame.draw.line(screen, (255,255,0), (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
        pygame.draw.line(screen, (0,255,0), (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
        pygame.draw.line(screen, (255,255,0), (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
        pygame.draw.line(screen, (0,255,0), (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
