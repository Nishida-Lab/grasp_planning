# python library
import numpy as np

# chainer library
import chainer
import chainer.functions as F
import chainer.links as L


def data_separator(x):
    len_x = x.shape[0]

    image = []
    rec = []

    for i in range(x.shape[0]):
        image.append(x[i][0].data)
        rec.append(x[i][1].data)
    image = np.asarray(image)
    rec = np.asarray(rec)

    image = chainer.Variable(image)
    rec = chainer.Variable(rec)
    return image,rec


def data_merger(cnn_out,rec):
    merged = []
    for i in range(cnn_out.shape[0]):
        merged.append(cnn_out[i].data.tolist()+rec[i].data.tolist())
    merged = np.asarray(merged).astype(np.float32)
    #print merged.shape
    #merged = chainer.Variable(merged)
    return merged


#Network definition
class CNN(chainer.Chain):

    def __init__(self, train= True):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(3, 3, 5),
            conv2 = L.Convolution2D(3, 3, 5),
            l1 = L.Linear(2997, 500),
            l2 = L.Linear(500, 200),
            l3 = L.Linear(208, 100),
            l4 = L.Linear(100, 2),
        )
        self.train = train

    def __call__(self, x):

        image,rec = data_separator(x)
        h = F.max_pooling_2d(self.conv1(image),2,stride = 2)
        h = F.max_pooling_2d(self.conv2(h),2,stride = 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = data_merger(h,rec)
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        return h
