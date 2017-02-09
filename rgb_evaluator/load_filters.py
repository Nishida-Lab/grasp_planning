#coding: utf-8
#python library
import cPickle
import matplotlib.pyplot as plt

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers

#python script
import network_structure as nn


if __name__ == '__main__':

    #model = cPickle.load(open("cnn03a.state", "rb"))

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    # 1つめのConvolution層の重みを可視化
    print model

    n1, n2, h, w = model.conv1.W.shape

    fig = plt.figure()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(n1):
        ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(model.conv1.W[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')

    plt.show()
