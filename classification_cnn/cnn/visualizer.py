#python library
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    plt.figure(1)
    plt.plot(epoch,train_loss,"b",label = "train LOSS")
    plt.plot(epoch,test_loss,"g",label = "test LOSS")
    plt.yscale('log')
    plt.title("LOSS reduction")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("LOSS")

    plt.figure(2)
    plt.plot(epoch,train_accuracy,"b",label = "train accuracy")
    plt.plot(epoch,test_accuracy,"g",label = "test accuracy ")
    plt.title("accuracy increase")
    plt.legend(loc = "lower right")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")


#separate dataset into input and output
def dataset_separator(dataset):

    data = []
    label = []

    for i in range(len(dataset)):
        data.append(dataset[i][0])
        label.append(dataset[i][1])

    data = np.array(data,dtype = np.float32)
    label = np.array(label,dtype = np.float32)

    return data,label


#visualize classification result
def result_visualizer(model,data,label):

    fig3 = plt.figure(3)
    plt.gray()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    p = np.random.random_integers(0, len(data),26)

    for i in range(0,25):

        data_i = data[p[i]]
        data_i = data_i.reshape(28,28)
        estimated_value = model.predictor(data_i.reshape(1,1,28,28)).data
        estimated_label_i = np.argmax(estimated_value)

        print 'estimated_label:' + str(estimated_label_i) + ' test_label:' + str(label[p[i]])

        ax = fig3.add_subplot(5, 5, i+1)
        ax.set_xlim(0, 28)
        ax.set_ylim(0, 28)
        ax.tick_params(labelbottom="off")
        ax.tick_params(labelleft="off")
        ax.set_title('%i' % estimated_label_i)
        xx = np.array(range(28 + 1))
        yy = np.array(range(28, -1, -1))
        X, Y = np.meshgrid(xx, yy)
        ax.pcolor(X,Y,data_i)


#visualize estimation error
def error_viualiser(model,test_data,test_label):

    error_data = []
    error_label = []
    error_counter = 0

    fig4 = plt.figure(figsize = plt.figaspect(0.5))
    plt.gray()
    plt.subplots_adjust(wspace=0.2)

    for i in range(len(test_data)):

        data_i = test_data[i]
        data_i = data_i.reshape(28,28)
        estimated_value = model.predictor(data_i.reshape(1,1,28,28)).data
        estimated_label_i = np.argmax(estimated_value)

        if estimated_label_i != test_label[i]:
            error_data.append(data_i)
            error_label.append([estimated_label_i,test_label[i]])
            error_counter = error_counter + 1

        if error_counter == 10:
            break

    for i in range(10):
        print 'estimated_label:' + str(error_label[i][0]) + ' test_label:' + str(error_label[i][1])

        ax = fig4.add_subplot(2, 5, i+1)
        ax.set_xlim(0, 28)
        ax.set_ylim(0, 28)
        ax.tick_params(labelbottom="off")
        ax.tick_params(labelleft="off")
        ax.set_title(str(error_label[i][1])+' != '+str(error_label[i][0]))
        #ax.set_title('%i != %i' % error_label[i][0], error_label[i][1])
        xx = np.array(range(28 + 1))
        yy = np.array(range(28, -1, -1))
        X, Y = np.meshgrid(xx, yy)
        ax.pcolor(X,Y,error_data[i])


#main
if __name__ == '__main__':
    loss_visualizer()
    plt.show()
