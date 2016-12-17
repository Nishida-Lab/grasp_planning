# python library
import numpy as np

# chainer library
import chainer
import chainer.functions as F
import chainer.links as L


# separate input data into image and rec
def data_separator(X):

    Xs_1 = X.shape[0]
    Xs_2 = X.shape[1]

    img = []
    rec = []

    #print Xs_1
    for n in range(Xs_1):
        rec.append(X.data[n][0:8])
        img.append(X.data[n][8:Xs_2])

    rec = np.asarray(rec).reshape(Xs_1,8).astype(np.float32)
    img = np.asarray(img).reshape(Xs_1,3,160,120).astype(np.float32)

    img = chainer.Variable(img)
    rec = chainer.Variable(rec)

    return img,rec


# merge cnn output and rec data
def data_merger(cnn_out,rec):
    merged = []
    for i in range(cnn_out.shape[0]):
        merged.append(cnn_out[i].data.tolist()+rec[i].data.tolist())
    merged = np.asarray(merged).astype(np.float32)
    #print merged.shape.astype(np.float32)
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
            l3 = L.Linear(200, 100),
            l4 = L.Linear(100, 2),
        )

    def __call__(self, x):
        img,rec = data_separator(x)
        h = F.max_pooling_2d(self.conv1(img),2,stride = 2)
        h = F.max_pooling_2d(self.conv2(h),2,stride = 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        #h = data_merger(h,rec)
        h = F.relu(self.l3(h))
        h = F.sigmoid(self.l4(h))
        return h
