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


def random_input(data_label):

    scale = 4

    theta = random.uniform(-1,1) * np.pi

    xc_yc = []

    xc = randint(1,680)
    yc = randint(1,480)

    xc_yc.append(xc)
    xc_yc.append(yc)


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
    return x,xc_yc,theta,rec_list


# write text
def captions():

    rec_a = np.array(rec_d)*4
    rec_b = []

    for i in range(8):
        rec_b.append(round(rec_a[i],2))

    text1 = font1.render("directory_n: "+str(directory_n)+" picture_n: "+str(picture_n), True, (255,255,255))
    text2 = font1.render("quit: ESC", True, (255,255,255))
    #text3 = font1.render("renew: z", True, (255,255,255))
    text4 = font2.render("rectangle:", True, (255,0,0))
    text5 = font2.render("  "+str(rec_b), True, (255,0,0))
    text6 = font2.render("center_point: "+str(center)+",  angle [deg]: "+str(round(angle*(180/np.pi),2)), True, (255,0,0))
    screen.blit(text1, [20, 20])
    screen.blit(text2, [20, 50])
    #screen.blit(text3, [20, 80])
    screen.blit(text4, [20, 370])
    screen.blit(text5, [20, 400])
    screen.blit(text6, [20, 440])


#main
if __name__ == '__main__':

    directory_n = 3
    picture_n = 50
    scale = 4

    data_label = label_handling(directory_n,picture_n)
    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    screen_size = (640, 480)
    pygame.init()
    pygame.display.set_mode(screen_size)
    pygame.display.set_caption("random planning")
    font1 = pygame.font.Font(None, 30)
    font2 = pygame.font.Font(None, 30)

    screen = pygame.display.get_surface()
    bg = pygame.image.load(path).convert_alpha()
    rect_bg = bg.get_rect()


    while (1):

        x,center,angle,rec_d = random_input(data_label)

        test_output = model.forward(chainer.Variable(x))
        test_label = np.argmax(test_output.data[0])

        print test_output.data
        print test_label
        print x[0:8]
        captions()

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
        captions()

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
