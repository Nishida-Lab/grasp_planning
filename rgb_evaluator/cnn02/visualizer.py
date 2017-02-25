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

import json


#visualize loss reduction
def loss_visualizer():

    epoch = []
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    f = open('./result/log', 'r') #load log file
    data = json.load(f)
    f.close()

    value = []

    for i in range(0,len(data)):
        value = data[i]
        epoch.append(value["epoch"])
        train_loss.append(value["main/loss"])
        test_loss.append(value["validation/main/loss"])
        train_accuracy.append(value["main/accuracy"])
        test_accuracy.append(value["validation/main/accuracy"])

    #fig1 = plt.figure(1)
    fig1 = plt.figure(1,figsize=(8,6))
    plt.plot(epoch,train_loss,"b",linewidth=2,label = "train LOSS")
    plt.plot(epoch,test_loss,"g",linewidth=2,label = "validation LOSS")
    plt.yscale('log')
    plt.grid(which="both")
    #plt.title("LOSS reduction")
    plt.legend(fontsize=20) #18
    plt.tick_params(labelsize=22) #18
    plt.xlabel("epoch",fontname='roman', fontsize=26)
    plt.ylabel("LOSS",fontname='roman', fontsize=26)
    fig1.subplots_adjust(left=0.15,bottom=0.15)
    ax = fig1.add_subplot(111)

    #fig2 = plt.figure(2)
    fig2 = plt.figure(2,figsize=(8,6))
    plt.plot(epoch,train_accuracy,"b",linewidth=2,label = "train accuracy")
    plt.plot(epoch,test_accuracy,"g",linewidth=2,label = "validation accuracy ")
    #plt.title("accuracy increase")
    plt.legend(loc = "lower right",fontsize=20)
    plt.tick_params(labelsize=22)
    plt.xlabel("epoch",fontname='roman',fontsize=26)
    plt.ylabel("accuracy",fontname='roman',fontsize=26)
    plt.yticks([i*0.1 for i in range(5,10,1)])
    fig2.subplots_adjust(left=0.15,bottom=0.15)
    ax = fig2.add_subplot(111)


#main
if __name__ == '__main__':
    loss_visualizer()
    plt.show()
