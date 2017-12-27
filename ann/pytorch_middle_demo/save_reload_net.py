#coding: utf-8
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    net = torch.nn.Sequential(
        torch.nn.Linear(1,30),
        torch.nn.ReLU(),
        torch.nn.Linear(30,30),
        torch.nn.ReLU(),
        torch.nn.Linear(30,1)
    )

    optm = torch.optim.SGD(net.parameters(),lr=5e-2)

    loss_func = torch.nn.MSELoss()

    for i in range(600):
        pred = net(x)
        loss = loss_func(pred,y)
        optm.zero_grad()
        loss.backward()
        optm.step()
    torch.save(net, 'net.pkl')  # 保存整个网络
    torch.save(net.state_dict(),'net_params.pkl')  # 只保存网络参数

def restore_net():
    net = torch.load('net.pkl')
    pred = net(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=5)
    plt.show()


def restore_params():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 1)
    )

    net.load_state_dict(torch.load('net_params.pkl'))
    pred = net(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=5)
    plt.show()



if __name__ == '__main__':
    save()
    # restore_net()
    restore_params()