#python library
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers

#python script
import network_structure as nn
import visualizer as v


#main
if __name__ == '__main__':

    model = L.Classifier(nn.CNN())
    serializers.load_npz('cnn.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    test_data,test_label = v.dataset_separator(test)

    v.loss_visualizer()
    v.result_visualizer(model,test_data,test_label)
    print '--------------------------------------'
    v.error_viualiser(model,test_data,test_label)
    plt.show()
