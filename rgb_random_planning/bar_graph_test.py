#python library
import numpy as np
import matplotlib.pyplot as plt
import csv


if __name__ == '__main__':

    file_id = 'success.csv'
    file_path = 'data/'

    read_file = file_path + file_id

    f = open(read_file, 'rb')
    data = csv.reader(f)
    next(data)

    index = 3
    object_name = []
    success_rate = []

    for row in data:
        object_name.append(row[0])
        success_rate.append(float(row[index]))

    x = [i for i in range(len(object_name))]

    fig1 = plt.figure(1,figsize=(8,12))
    ax1 = fig1.add_subplot(111)
    plt.barh(x,success_rate,align="center")
    plt.yticks(x, object_name,fontname='roman',fontsize=18)
    ax1.set_xlabel('Success rate [%]',fontname='roman',fontsize=18)
    ax1.set_xlim([0,100])
    ax1.set_ylim([32,-1])
    ax1.set_xticks([i for i in range(0,110,10)])
    ax1.set_yticks([i for i in range(len(object_name))])
    ax1.tick_params(labelsize=18)
    #ax2 = ax1.twiny()
    #ax2.set_xlim(ax1.get_xlim())
    #ax2.set_xticks([i for i in range(0,110,10)])
    #ax2.set_xlabel('Success rate [%]',fontname='roman',fontsize=18)
    #ax2.tick_params(labelsize=18)
    #plt.xlabel('Success rate [%]',fontname='roman',fontsize=18)
    #fig1.subplots_adjust(bottom=0.1)
    fig1.subplots_adjust(left=0.3, wspace=0.4)

    fig2 = plt.figure(2,figsize=(10,6))
    plt.bar(x,success_rate,align="center")
    plt.xticks(x, object_name,fontname='roman',rotation=90)
    plt.xlim([-1,32])
    plt.ylim([0,100])
    plt.xticks([i for i in range(len(object_name))])
    plt.yticks([i for i in range(0,110,10)])
    plt.tick_params(labelsize=18)
    plt.ylabel('Success rate [%]',fontname='roman',fontsize=18)
    fig2.subplots_adjust(bottom=0.5)

    plt.show()
