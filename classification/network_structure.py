#chainer library
import chainer
import chainer.functions as F
import chainer.links as L

#Network definition
class CNN(chainer.Chain):

    def __init__(self, train= True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            fc=L.Linear(800, 500),
            out=L.Linear(500, 10),
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)),2,stride = 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)),2,stride = 2)
        h3 = self.fc(F.dropout(h2))
        return self.out(h3)
