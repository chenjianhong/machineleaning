#coding: utf-8
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append('../..')
import matplotlib.pyplot as plt

class RegreNN(torch.nn.Module):
    
    def __init__(self, input_feature, output_feature):
        super(RegreNN, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_feature,30)
        self.predict_layer = torch.nn.Linear(30, output_feature)
    
    def forward(self, x):
        h = self.hidden_layer(x)
        m = F.relu(h)
        o = self.predict_layer(m)
        return o

def main():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())

    x = Variable(x)
    y = Variable(y)

    net = RegreNN(1,1)
    optm = torch.optim.SGD(net.parameters(),lr=0.5e-1)
    loss_func = torch.nn.MSELoss()

    plt.ion()
    for i in range(600):
        v = net(x)
        loss = loss_func(v,y)
        optm.zero_grad()
        loss.backward()
        optm.step()

        if i % 100 == 0:
            print(loss)
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), v.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()