#python library
import argparse
import time
import numpy as np

#chainer libraty
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers

from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer.functions.loss.mean_squared_error import mean_squared_error
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy

#python scripts
import get_dataset as d
import network_structure as nn


#main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GRASP CLASSIFER')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # if args.gpu >= 0:
    #     chainer.cuda.get_device(args.gpu).use()
    #     model.to_gpu()
    # print 'GPU: ' + format(args.gpu)

    print 'minibatch-size: ' + format(args.batchsize)
    print 'epoch: ' + format(args.epoch)
    print ''

    # load CNN structure
    model = nn.CNN_classification()
    #model = nn.CNN_classification1()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    #optimizer = chainer.optimizers.Adam(alpha=0.00001)
    optimizer.setup(model)

    # Load grasp dataset
    scale = 2
    train_N = 4800
    validation_N = 400
    test_N = 10
    Xt,Yt,Xv,Yv = d.generate_dataset(train_N,validation_N,scale)
    train = zip(Xt,Yt)
    test = zip(Xv,Yv)

    start_time = time.time() #start time measurement
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    execution_time = time.time() - start_time

    print "execution time : " + str(execution_time)

    serializers.save_npz('cnn.model', model)
    serializers.save_npz('cnn.state', optimizer)
