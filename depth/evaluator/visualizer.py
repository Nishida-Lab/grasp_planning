# python library
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt

import json


# separate input data into image and rec
def data_separator(X):

    img = []
    rec = []

    for i in range(len(X)):
        if i < 8:
            rec.append(X[i])
        else:
            img.append(X[i])

    rec = np.asarray(rec).reshape(8,1).astype(np.float32)
    img = np.asarray(img).reshape(1,165,200).astype(np.float32)

    return img,rec


# draw rectangle
def draw_rec(x,estimated,actual,scale):

    img,rec = data_separator(x)

    img_array = np.array(img).reshape(330/scale,400/scale)
    img_array = img_array*255

    image = Image.fromarray(np.uint8(img_array))
    image = image.convert("RGB")
    resized_img = image.resize((400,330))

    rec = rec*scale
    draw_rec = ImageDraw.Draw(resized_img)

    if actual == 0:
        rec_color = ['red','blue']
        # draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='red', width=2)
        # draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='blue', width=2)
        # draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
        # draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
        actual_label = 'negative'
    elif actual == 1:
        rec_color = ['yellow','green']
        # draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
        # draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
        # draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='yellow', width=2)
        # draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='green', width=2)
        actual_label = 'positive'

    draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill=rec_color[0], width=2)
    draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill=rec_color[1], width=2)
    draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill=rec_color[0], width=2)
    draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill=rec_color[1], width=2)

    #set image label
    if estimated == 0:
        estimated_label = 'negative'
    else:
        estimated_label = 'positive'

    image_label1 = " estimated: " + estimated_label
    image_label2 = " actual: " + actual_label
    draw_rec.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
    draw_rec.text((10,280), image_label1, (255, 0, 0))
    draw_rec.text((10,300), image_label2, (255, 0, 0))

    resized_img.show()


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

    fig1 = plt.figure(1,figsize=(8,6))
    plt.plot(epoch,train_loss,"b",linewidth=2,label = "train LOSS")
    plt.plot(epoch,test_loss,"g",linewidth=2,label = "validation LOSS")
    plt.yscale('log')
    #plt.title("LOSS reduction")
    plt.legend(fontsize=18)
    plt.xlabel("epoch",fontname='roman', fontsize=22)
    plt.ylabel("LOSS",fontname='roman', fontsize=22)
    plt.tick_params(labelsize=18)
    fig1.subplots_adjust(bottom=0.15)
    ax = fig1.add_subplot(111)

    fig2 = plt.figure(2,figsize=(8,6))
    plt.plot(epoch,train_accuracy,"b",linewidth=2,label = "train accuracy")
    plt.plot(epoch,test_accuracy,"g",linewidth=2,label = "validation accuracy ")
    #plt.title("accuracy increase")
    plt.legend(loc = "lower right",fontsize=18)
    plt.xlabel("epoch",fontname='roman',fontsize=22)
    plt.ylabel("accuracy",fontname='roman',fontsize=22)
    plt.yticks([i*0.1 for i in range(5,10,1)])
    plt.tick_params(labelsize=18)
    fig2.subplots_adjust(bottom=0.15)
    ax = fig2.add_subplot(111)


#main
if __name__ == '__main__':
    loss_visualizer()
    plt.show()
