#python library
import numpy as np
import matplotlib.pyplot as plt
import csv


def drawer7(n,r,k):

    x = [i for i in range(7)]

    fig = plt.figure(k,figsize=(8,6))
    ax1 = fig.add_subplot(111)
    plt.barh(x,r,align="center")
    ax1.set_xlim([0,100])
    ax1.set_ylim([7,-1])
    ax1.set_xticks([i for i in range(0,110,20)])
    ax1.set_yticks([i for i in range(len(n))])
    ax1.tick_params(labelsize=24)
    plt.yticks(x,n,fontname='roman',fontsize=26)
    ax1.set_xlabel('success rate [%]',fontname='roman',fontsize=26)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([i for i in range(0,110,10)])
    ax2.set_xticklabels([])
    plt.grid()
    fig.subplots_adjust(left=0.35,bottom=0.15)


# 7 data
def bar_graph_7(name,rate):

    index = np.argsort(rate)[::-1][:len(rate)]
    success_rate = np.sort(rate)[::-1][:len(rate)]
    object_name = []

    for i in index:
        object_name.append(name[i])

    for i in range(3):
        c = i*7
        drawer7(object_name[c:c+7],success_rate[c:c+7],i)


# all data
def bar_graph_all(name,rate):

    index = np.argsort(rate)[::-1][:len(rate)]
    success_rate = np.sort(rate)[::-1][:len(rate)]
    object_name = []

    for i in index:
        object_name.append(name[i])

    x = [i for i in range(len(object_name))]

    fig = plt.figure(10,figsize=(10,14))
    ax1 = fig.add_subplot(111)
    plt.barh(x,success_rate,align="center")
    plt.yticks(x, object_name,fontname='roman',fontsize=18)
    ax1.set_xlabel('success rate [%]',fontname='roman',fontsize=21)
    ax1.set_xlim([0,100])
    ax1.set_ylim([21,-1])
    ax1.set_xticks([i for i in range(0,110,10)])
    ax1.set_yticks([i for i in range(len(object_name))])
    ax1.tick_params(labelsize=18)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([i for i in range(0,110,10)])
    ax2.set_xticklabels([])
    plt.grid()
    fig.subplots_adjust(left=0.25,bottom=0.15)


#main
if __name__ == '__main__':

    file_id = 'result.csv'
    file_path = 'data/'

    read_file = file_path + file_id

    f = open(read_file, 'rb')
    data = csv.reader(f)
    next(data)

    index = 4
    name = []
    rate = []

    for row in data:
        #print row
        name.append(row[0])
        rate.append(float(row[index]))

    bar_graph_all(name,rate)
    bar_graph_7(name,rate)

    plt.show()
