#coding:utf-8
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class ClafNN(torch.nn.Module):

    def __init__(self,input_feature,output_feature):
        super(ClafNN, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_feature,10)
        self.output_layer = torch.nn.Linear(10,output_feature)

    def forward(self, x):
        h = self.hidden_layer(x)
        ac = F.relu(h)
        o = self.output_layer(ac)
        return o

def main():
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer
    x, y = Variable(x), Variable(y)
    print(y.size())

    net = ClafNN(2,2)

    loss_func = torch.nn.CrossEntropyLoss()

    optm = torch.optim.SGD(net.parameters(),lr=0.01)

    plt.ion()

    for i in range(100):

        o = net(x)

        loss = loss_func(o,y)

        optm.zero_grad()
        loss.backward()
        optm.step()

        if i % 10 ==0:
            plt.cla()
            prediction = torch.max(F.softmax(o),1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y) / 200.
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()