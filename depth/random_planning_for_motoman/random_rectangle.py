#python library
import numpy as np
from numpy.random import *
import random


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
