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

    validation_N = 100

    model = nn.CNN_classification3()
    serializers.load_npz('cnn03a.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    Xv,Yv = d.validation_dataset(validation_N)
    validation_output = model.forward(chainer.Variable(Xv))

    print "< negative = 0, positive = 1 >"
    success_n = 0
    estimated = []
    actual = []
    for i in range(validation_N):

        validation_label = np.argmax(validation_output.data[i])

        if validation_label == 0:
            estimated.append(0)
        else:
            estimated.append(1)

        if Yv[i] == 0:
            actual.append(0)
        else:
            actual.append(1)

        if estimated[i] == actual[i]:
            success_n += 1.0

        print "validation_n: " +str(i)+ " estimated: " + str(estimated[i]) + " actual: " + str(actual[i])

    ac = float(success_n/validation_N)
    print "accuracy: " + str(ac)

    for i in range(3):
        v.draw_rec(Xv[i],estimated[i],actual[i])

    v.loss_visualizer()
    plt.show()
