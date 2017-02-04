# python library
import numpy as np

# chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter


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
    img = np.asarray(img).reshape(Xs_1,3,120,160).astype(np.float32)
    #print rec

    img = chainer.Variable(img)
    rec = chainer.Variable(rec)

    return img,rec


# merge cnn output and rec data
def data_merger(cnn_out,rec):
    merged = []
    for i in range(cnn_out.shape[0]):
        merged.append(cnn_out[i].data.tolist()+rec[i].data.tolist())
    merged = np.asarray(merged,dtype=np.float32)
    #print merged[0].shape
    #merged = chainer.Variable(merged)
    return merged


# pick up teacher data [0,1]
def pick_up_t(t_data):
    t = []
    for i in range(len(t_data)):
        t.append(t_data[i])
    t = np.array(t,dtype=np.int32)
    return t


class CNN_classification1(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_classification1, self).__init__(
            conv = L.Convolution2D(3, 3, 5),
            li1 = L.Linear(2808, 500),
            li2 = L.Linear(500, 200),
            li3 = L.Linear(200, 50),
            lr1 = L.Linear(8,50),
            #lo1 = L.Linear(50, 50), #just add
            lo1 = L.Linear(100, 50), #concat
            lo2 = L.Linear(50, 2),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        img,rec = data_separator(x)
        hi = self.conv(img)
        hi = self.conv(hi)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = self.conv(hi)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = F.relu(self.li1(hi))
        hi = F.relu(self.li2(hi))
        hi = F.relu(self.li3(hi))
        hr = F.relu(self.lr1(rec))
        #h = hi + hr
        h = F.concat((hi,hr),axis=1)
        h = F.relu(self.lo1(F.dropout(h)))
        h = self.lo2(h)
        return h

    def __call__(self, x, t):
        self.clear()
        h = self.forward(x)
        #print h.data[0]
        #print t.data
        to = pick_up_t(t.data)
        self.loss = F.softmax_cross_entropy(h, to)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(h, to)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss



class CNN_classification2(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_classification2, self).__init__(
            conv = L.Convolution2D(3, 3, 5),
            li1 = L.Linear(2997, 200),
            li2 = L.Linear(200, 50),
            lr1 = L.Linear(8,50),
            lo1 = L.Linear(100, 50), #concat
            lo2 = L.Linear(50, 2),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        img,rec = data_separator(x)
        hi = self.conv(img)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = self.conv(hi)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = F.relu(self.li1(hi))
        hi = F.relu(self.li2(hi))
        hr = F.relu(self.lr1(rec))
        #h = hi + hr
        h = F.concat((hi,hr),axis=1)
        h = F.relu(self.lo1(F.dropout(h)))
        h = self.lo2(h)
        return h

    def __call__(self, x, t):
        self.clear()
        h = self.forward(x)
        #print h.data[0]
        #print t.data
        to = pick_up_t(t.data)
        self.loss = F.softmax_cross_entropy(h, to)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(h, to)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class CNN_classification3(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_classification3, self).__init__(
            conv = L.Convolution2D(3, 3, 5),
            li1 = L.Linear(2997, 100),
            li2 = L.Linear(100, 50),
            lr1 = L.Linear(8,50),
            lo1 = L.Linear(100, 30), #concat
            lo2 = L.Linear(30, 2),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        img,rec = data_separator(x)
        hi = self.conv(img)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = self.conv(hi)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = F.relu(self.li1(hi))
        hi = F.relu(self.li2(hi))
        hr = F.relu(self.lr1(rec))
        #h = hi + hr
        h = F.concat((hi,hr),axis=1)
        h = F.relu(self.lo1(F.dropout(h)))
        h = self.lo2(h)
        return h

    def __call__(self, x, t):
        self.clear()
        h = self.forward(x)
        #print h.data[0]
        #print t.data
        to = pick_up_t(t.data)
        self.loss = F.softmax_cross_entropy(h, to)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(h, to)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class CNN_classification4(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_classification4, self).__init__(
            conv = L.Convolution2D(3, 3, 5),
            li1 = L.Linear(13572, 100),
            li2 = L.Linear(100, 30),
            lr1 = L.Linear(8,30),
            lo1 = L.Linear(60, 30), #concat
            lo2 = L.Linear(30, 2),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        img,rec = data_separator(x)
        hi = self.conv(img)
        hi = F.max_pooling_2d(hi,2,stride = 2)
        hi = F.relu(self.li1(hi))
        hi = F.relu(self.li2(hi))
        hr = F.relu(self.lr1(rec))
        #h = hi + hr
        h = F.concat((hi,hr),axis=1)
        h = F.relu(self.lo1(F.dropout(h)))
        h = self.lo2(h)
        return h

    def __call__(self, x, t):
        self.clear()
        h = self.forward(x)
        #print h.data[0]
        #print t.data
        to = pick_up_t(t.data)
        self.loss = F.softmax_cross_entropy(h, to)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(h, to)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
