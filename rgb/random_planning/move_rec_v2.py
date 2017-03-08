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

    return img_list


def key_input(theta,xc,yc,a,b):

    scale = 4

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

    img_list = load_picture(data_label,4)
    x = rec_list + img_list
    x = np.array(x,dtype=np.float32).reshape((1,57608))

    return x,np.array(rec_list)*scale


#main
if __name__ == '__main__':

    #directly_n = 6
    #picture_n = 34
    directly_n = randint(1,8)
    picture_n = randint(0,99)
    scale = 4

    data_label = label_handling(directly_n,picture_n)
    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    pygame.font.init()
    font1 = pygame.font.Font(None, 30)
    font2 = pygame.font.Font(None, 25)

    screen_size = (640, 480)
    pygame.init()
    pygame.display.set_mode(screen_size)
    pygame.display.set_caption("move rectangle")

    screen = pygame.display.get_surface()
    bg = pygame.image.load(path).convert_alpha()
    rect_bg = bg.get_rect()

    theta = random.uniform(-1,1) * np.pi

    theta = 0
    xc = randint(120,520)
    yc = randint(120,360)
    a = randint(1,85)
    b = randint(1,60)

    while (1):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == K_z:
                    theta -= 5*np.pi/180
                if event.key == K_x:
                    theta += 5*np.pi/180

                if event.key == K_RIGHT:
                    xc += 20
                if event.key == K_LEFT:
                    xc -= 20

                if event.key == K_UP:
                    yc -= 10
                if event.key == K_DOWN:
                    yc += 10

                if event.key == K_a:
                    a += 5
                if event.key == K_s:
                    a -= 5

                if event.key == K_q:
                    b += 5
                if event.key == K_w:
                    b -= 5

        x,rec = key_input(theta,xc,yc,a,b)

        test_output = model.validation(chainer.Variable(x))
        test_label = np.argmax(test_output.data[0])

        #print test_label
        text1 = font1.render("move center point: UP,DOWN,LEFT,RIGHT", True, (255,255,255))
        text2 = font1.render("rotate + : Z, rotate - : X", True, (255,255,255))
        text3 = font1.render("width +  : A, width -  : S", True, (255,255,255))
        text4 = font1.render("height + : Q, height - : W", True, (255,255,255))
        text5 = font1.render("quit: ESC", True, (255,255,255))
        text6 = font2.render("rectangle = ", True, (255,0,0))
        text7 = font2.render(str(rec), True, (255,0,0))
        text8 = font1.render("directory_n: "+str(directly_n)+" picture_n: "+str(picture_n), True, (255,255,255))
        screen.blit(text1, [20, 20])
        screen.blit(text2, [20, 50])
        screen.blit(text3, [20, 80])
        screen.blit(text4, [20, 110])
        screen.blit(text5, [20, 140])
        screen.blit(text6, [20, 380])
        screen.blit(text7, [20, 400])
        screen.blit(text8, [20, 450])

        pygame.display.update()
        pygame.time.wait(30)
        screen.fill((0, 0, 0))
        screen.blit(bg, rect_bg)

        if test_label == 1:
            pygame.draw.line(screen, (255,255,0), (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
            pygame.draw.line(screen, (0,255,0), (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
            pygame.draw.line(screen, (255,255,0), (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
            pygame.draw.line(screen, (0,255,0), (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)
        else:
            pygame.draw.line(screen, (255,0,0), (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
            pygame.draw.line(screen, (0,0,255), (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
            pygame.draw.line(screen, (255,0,0), (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
            pygame.draw.line(screen, (0,0,255), (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)
