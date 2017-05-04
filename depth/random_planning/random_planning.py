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
import time

# OpenCV
import cv2

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


# show RGB image
def show_picture(path):
    path = p.data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]
    img = cv2.imread(path+'r.png')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.pause(0.05)


# load image data
def load_picture(path,scale):

    img = Image.open(path)

    resize_img = img.resize((img.size[0]/scale,img.size[1]/scale))

    img_array = np.asanyarray(resize_img,dtype=np.float32)
    img_shape = img_array.shape
    img_array = np.reshape(img_array,(img_shape[1]*img_shape[0],1))
    img_list = []
    for i in range(len(img_array)):
        img_list.append(img_array[i][0]/255.0)

    return img_list


# generate grasp rectangle randomly
def random_rec(object_area,rec_area,scale):

    theta = random.uniform(-1,1) * np.pi

    i = object_area.shape[0]

    while True:

        area_index = 0
        i = object_area.shape[0]

        if i==0:
            area_index = 0
        else:
            area_index = randint(0,i)

        N = len(object_area[area_index])

        i = randint(1,N)
        j = (i+N/2)%N
        k = (i+N/8)%N
        h = (i+(5*N/8))%N

        p1 = object_area[area_index][i]
        p2 = object_area[area_index][j]
        p3 = object_area[area_index][k]
        p4 = object_area[area_index][h]

        xc_yc = []

        s1 = ((p2[0][0]-p1[0][0])*(p3[0][1]-p1[0][1])-(p2[0][1]-p1[0][1])*(p3[0][0]-p1[0][0]))/2
        s2 = ((p2[0][0]-p1[0][0])*(p1[0][1]-p4[0][1])-(p2[0][1]-p1[0][1])*(p1[0][0]-p4[0][0]))/2

        xc = (p3[0][0]+(p4[0][0]-p3[0][0])*s1/(s1+s2))
        yc = (p3[0][1]+(p4[0][1]-p3[0][1])*s1/(s1+s2))

        if rec_area[area_index][0] < xc and xc < rec_area[area_index][0]+rec_area[area_index][2] \
           and rec_area[area_index][1] < yc and yc < rec_area[area_index][1]+rec_area[area_index][3]:
            break

    xc_yc.append(xc-100)
    xc_yc.append(yc-100)

    w = randint(20,80)
    h = 15

    x1 = w*np.cos(theta)-h*np.sin(theta)+xc
    y1 = w*np.sin(theta)+h*np.cos(theta)+yc

    x2 = -w*np.cos(theta)-h*np.sin(theta)+xc
    y2 = -w*np.sin(theta)+h*np.cos(theta)+yc

    x3 = -w*np.cos(theta)+h*np.sin(theta)+xc
    y3 = -w*np.sin(theta)-h*np.cos(theta)+yc

    x4 = w*np.cos(theta)+h*np.sin(theta)+xc
    y4 = w*np.sin(theta)-h*np.cos(theta)+yc

    rec_list = []

    rec_list.append(round((x1-100)/scale,2))
    rec_list.append(round((y1-100)/scale,2))

    rec_list.append(round((x2-100)/scale,2))
    rec_list.append(round((y2-100)/scale,2))

    rec_list.append(round((x3-100)/scale,2))
    rec_list.append(round((y3-100)/scale,2))

    rec_list.append(round((x4-100)/scale,2))
    rec_list.append(round((y4-100)/scale,2))

    return rec_list,xc_yc,theta,w,p1,p2,p3,p4


# generate input data for CNN
def input_data(path,rec_list,scale):
    img_list = load_picture(path,scale)
    x = rec_list + img_list
    x = np.array(x,dtype=np.float32).reshape((1,33008))

    return x


# calculate z
def calculate_z(path,rec,scale):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grayed)
    x1 = int(rec[1]*scale)
    y1 = int(rec[0]*scale)
    x2 = int(rec[5]*scale)
    y2 = int(rec[4]*scale)
    ml = np.max(img_array[np.min([x1,x2]):np.max([x1,x2]),np.min([y1,y2]):np.max([y1,y2])])
    z =round(ml*(150/255.0),2)
    return z


# update
def update_pygame(scr):
    pygame.display.update()
    #pygame.time.wait(100)
    scr.fill((0, 0, 0))
    scr.blit(bg, rect_bg)


# draw object area
def draw_object_rectangle(rec_area):
    for i in range(len(rec_area)):
        rec = (rec_area[i][0]-100, rec_area[i][1]-100, rec_area[i][2], rec_area[i][3])
        pygame.draw.rect(screen, (120,120,255), Rect(rec),3)


# draw grasp rectangle
def draw_grasp_rectangle(color1,color2):
    pygame.draw.line(screen, rec_color1, (x[0][0]*scale,x[0][1]*scale), (x[0][2]*scale,x[0][3]*scale),5)
    pygame.draw.line(screen, rec_color2, (x[0][2]*scale,x[0][3]*scale), (x[0][4]*scale,x[0][5]*scale),5)
    pygame.draw.line(screen, rec_color1, (x[0][4]*scale,x[0][5]*scale), (x[0][6]*scale,x[0][7]*scale),5)
    pygame.draw.line(screen, rec_color2, (x[0][6]*scale,x[0][7]*scale), (x[0][0]*scale,x[0][1]*scale),5)


# write text
def captions(dir_n,pic_n,f1):
    text1 = f1.render("directory_n: "+str(dir_n)+"   picture_n: "+str(pic_n), True, (255,255,255))
    text2 = f1.render("quit: ESC", True, (255,255,255))
    text3 = f1.render("renew: z", True, (255,255,255))
    screen.blit(text1, [20, 15])
    screen.blit(text2, [20, 50])
    screen.blit(text3, [20, 80])


#main
if __name__ == '__main__':


    directory_n = input('Directory No > ')
    picture_n = input('Image No > ')

    # random checking
    #directory_n = randint(9)+1
    #picture_n = randint(40)+1

    # multiple object recrangles will be appeard
    #directory_n = 7
    #picture_n = 80

    # "yellow plate"
    #directory_n = 3
    #picture_n = 88

    # white object
    #directory_n = 2
    #picture_n = 5

    scale = 2

    print 'directory:'+str(directory_n)+' picture:'+str(picture_n)
    data_label = label_handling(directory_n,picture_n)

    show_picture(data_label)

    path = p.data_path()+data_label[0]+'/dp'+data_label[0]+data_label[1]+'r.png'

    model = nn.CNN_classification()
    serializers.load_npz('cnn.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    screen_size = (400, 330)
    pygame.init()
    pygame.display.set_mode(screen_size)
    pygame.display.set_caption("random planning")

    screen = pygame.display.get_surface()
    bg = pygame.image.load(path).convert_alpha()
    rect_bg = bg.get_rect()

    pygame.font.init()
    font1 = pygame.font.Font(None, 30)

    search_area,rec_area = po.find_object_from_RGB(data_label)


    while (1):

        start = time.time()
        update_pygame(screen)
        rec,center,angle,w,p1,p2,p3,p4 = random_rec(search_area,rec_area,scale)
        x = input_data(path,rec,scale)
        test_output = model.forward(chainer.Variable(x))
        test_label = np.argmax(test_output.data[0])

        # draw grasp rectangle
        if test_label == 1:
            rec_color1 = (255,255,0)
            rec_color2 = (0,255,0)
            replay = 1

            # display xc_yc,theta
            angle = round(angle*180/np.pi,2)
            zc = calculate_z(path,rec,scale)
            print '(xc,yc): '+str(center)+',  zc[mm]: '+str(zc)
            print 'theta[deg]: '+str(angle)+',  gripper_width: '+str(w)+'\n'

            while(replay==1):

                update_pygame(screen)
                draw_object_rectangle(rec_area)
                draw_grasp_rectangle(rec_color1,rec_color2)

                # pygame.draw.circle(screen, (255,0,0), (p1[0][0]-100, p1[0][1]-100), 5)
                # pygame.draw.circle(screen, (0,255,0), (p2[0][0]-100, p2[0][1]-100), 5)
                # pygame.draw.circle(screen, (0,0,255), (p3[0][0]-100, p3[0][1]-100), 5)
                # pygame.draw.circle(screen, (0,0,0), (p4[0][0]-100, p4[0][1]-100), 5)
                pygame.draw.circle(screen, (255,255,0), (center[0],center[1]), 5)
                # pygame.draw.circle(screen, (255,0,0), (int(rec[0]*scale),int(rec[1]*scale)), 5)
                # pygame.draw.circle(screen, (255,0,0), (int(rec[4]*scale),int(rec[5]*scale)), 5)


                captions(directory_n,picture_n,font1)
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_z:
                            replay = 0
                        if event.key == K_ESCAPE:
                            pygame.quit()
                            sys.exit()
        else:
            update_pygame(screen)
            captions(directory_n,picture_n,font1)
