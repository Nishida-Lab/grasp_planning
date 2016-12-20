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
import get_dataset as d
import visualizer as v


#main
if __name__ == '__main__':

    test_N = 10

    model = nn.CNN_classification()
    serializers.load_npz('cnn.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    Xtest,Ytest = d.test_dataset(test_N)
    test_output = model.forward(chainer.Variable(Xtest))

    print "< negative = 0, positive = 1 >"
    success_n = 0
    estimated = []
    actual = []
    for i in range(test_N):

        test_label = np.argmax(test_output.data[i])

        if test_label == 0:
            estimated.append(0)
        else:
            estimated.append(1)

        if Ytest[i] == 0:
            actual.append(0)
        else:
            actual.append(1)

        if estimated[i] == actual[i]:
            success_n += 1.0

        print "test_n: " +str(i)+ " estimated: " + str(estimated[i]) + " actual: " + str(actual[i])

    ac = float(success_n/test_N)
    print "accuracy: " + str(ac)

    for i in range(3):
        v.draw_rec(Xtest[i],estimated[i],actual[i])

    v.loss_visualizer()
    plt.show()
